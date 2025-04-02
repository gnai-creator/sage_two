# --- predictor.py ---
from cog import BasePredictor, Input
from typing import List, Dict, Any
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import load_model

from model_trainable import SageDialogueModel

class Predictor(BasePredictor):
    def setup(self):
        self.model = SageDialogueModel()
        self.model.build((None, 10, 384))
        
        # Carrega o decoder para intenção
        self.decoder = load_model("decoder_model.h5")
        with open("index_to_word.json", "r", encoding="utf-8") as f:
            self.index_to_word = json.load(f)

        try:
            self.model.load_weights("sage_trained.weights.h5")
            print("Pesos carregados com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar pesos: {e}")
        
        print("Modelo carregado.")

    def decode_embedding(self, embedding):
        x = np.array(embedding, dtype=np.float32).reshape(1, -1)  # (1, 96)
        preds = self.decoder.predict(x)
        token_ids = np.atleast_1d(np.argmax(preds, axis=-1)[0]).tolist()
        words = [self.index_to_word.get(str(i), "<UNK>") for i in token_ids]
        return " ".join(words).strip()


    def predict(
        self,
        sequence: str = Input(description="Sequência JSON com shape [batch, 10, 384] ou apenas [384]"),
        reset: bool = Input(default=False, description="Reinicia o estado simbólico se True")
    ) -> Any:
        if reset:
            self.model.sage.reset_state()

        # Decodifica a sequência
        parsed = json.loads(sequence)

        # Caso receba um vetor plano [384], converte para [1, 10, 384]
        if isinstance(parsed[0], (float, int)):
            parsed = [parsed] * 10  # shape [10, 384]
        elif isinstance(parsed[0], list) and len(parsed) == 384:
            parsed = [parsed] * 10

        # Garante que seja [1, 10, 384]
        input_array = np.array(parsed, dtype=np.float32).reshape(1, 10, 384)
        input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)

        # Chamada do modelo
        output = self.model(input_tensor, training=False)
        predictions = output.numpy().flatten().tolist()
        symbolic_state = self.model.sage.get_symbolic_state()
        intention_text = self.decode_embedding(symbolic_state)

        # Fallbacks para garantir saída válida
        predictions = predictions if predictions else [0.0] * 384
        symbolic_state = symbolic_state if symbolic_state else [0.0] * 96
        intention_text = intention_text if intention_text.strip() else "<intenção vazia>"

        resultado = {
            "output": predictions if predictions else [0.0] * 384,
            "symbolic_state": symbolic_state if symbolic_state else [0.0] * 96,
            "intention": intention_text if intention_text.strip() else "<intenção vazia>"
        }

        print("⚙️ Output final:", json.dumps(resultado, indent=2, ensure_ascii=False))
        return resultado

