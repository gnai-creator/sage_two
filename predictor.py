import tensorflow as tf
import numpy as np


from tensorflow.keras.layers import (
    Dense, LayerNormalization,
    MultiHeadAttention, GlobalMaxPooling1D,
    GlobalAveragePooling1D
)


class SAGE_TWO(tf.keras.layers.Layer):
    def __init__(self,
                 memory_units=96,
                 symbolic_units=96,
                 num_heads=4,
                 dropout_rate=0.1,
                 num_iterations=1,
                 use_attention=True,
                 use_symbolic=True,
                 use_positional_encoding=True,
                 cross_attention=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.memory_units = memory_units
        self.symbolic_units = symbolic_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.use_attention = use_attention
        self.use_symbolic = use_symbolic
        self.use_positional_encoding = use_positional_encoding
        self.cross_attention = cross_attention

        if self.use_attention:
            self.attention = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.memory_units // self.num_heads,
                dropout=self.dropout_rate
            )
            self.temporal_norm = LayerNormalization()

        if self.use_symbolic:
            self.project_to_symbolic = Dense(self.symbolic_units)
            self.symbolic_mlp = tf.keras.Sequential([
                Dense(self.symbolic_units, activation='gelu'),
                Dense(self.symbolic_units)
            ])
            self.symbolic_norm = LayerNormalization()

        if self.cross_attention:
            self.cross_attn = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.symbolic_units // self.num_heads,
                dropout=self.dropout_rate
            )
            self.cross_norm = LayerNormalization()

        self.pooling_avg = GlobalAveragePooling1D()
        self.pooling_max = GlobalMaxPooling1D()

        self.head_dense = tf.keras.Sequential([
            Dense(128, activation='gelu'),
            Dense(64, activation='gelu')
        ])

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.seq_len = input_shape[1]

        self.memory_kernel = self.add_weight(
            name="memory_kernel",
            shape=(self.input_dim, self.memory_units),
            initializer="he_normal",
            trainable=True
        )

        if self.use_positional_encoding:
            self.pos_encoding = self.add_weight(
                name="pos_encoding",
                shape=(self.seq_len, self.input_dim),
                initializer="random_normal",
                trainable=True
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs

        if self.use_positional_encoding:
            pos_encoded = x + self.pos_encoding[tf.newaxis, :, :]
        else:
            pos_encoded = x

        memory_proj = tf.einsum('bsi,ij->bsj', pos_encoded, self.memory_kernel)

        if self.use_attention:
            attended = self.attention(
                query=memory_proj,
                value=memory_proj,
                key=memory_proj,
                training=training
            )
            attended = self.temporal_norm(attended + memory_proj)
        else:
            attended = memory_proj

        if self.use_symbolic:
            symbolic = self.project_to_symbolic(attended)
            for _ in range(self.num_iterations):
                refined = self.symbolic_mlp(symbolic, training=training)
                symbolic = self.symbolic_norm(symbolic + refined)
        else:
            symbolic = attended

        if self.cross_attention:
            cross = self.cross_attn(
                symbolic, symbolic, symbolic, training=training)
            symbolic = self.cross_norm(symbolic + cross)

        pooled_avg = self.pooling_avg(symbolic)
        pooled_max = self.pooling_max(symbolic)
        pooled = tf.concat([pooled_avg, pooled_max], axis=-1)

        features = self.head_dense(pooled)
        return features


class Predictor:
    def __init__(self):
        # Instancia o modelo diretamente
        self.model = SAGE_TWO()
        # Build do modelo (com shape genérico) para inicializar pesos
        # [batch=1, seq_len=10, features=16]
        dummy_input = tf.zeros((1, 10, 16))
        _ = self.model(dummy_input, training=False)

    def predict(self, inputs):
        # Espera: {"sequence": [[[...], [...], ...]]}
        x = np.array(inputs["sequence"], dtype=np.float32)
        x_tensor = tf.convert_to_tensor(x)
        output = self.model(x_tensor, training=False)
        return {"output": output.numpy().tolist()}
