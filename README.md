# SAGE_TWO - Replicate Model

Camada simbólica neural com atenção iterativa, projeção para espaço simbólico e pooling global. Parte da série SAGE.

## Como usar

Este modelo espera uma entrada no formato:

```json
{
  "sequence": [[[0.1, 0.2, ...], ..., [0.9, 0.8, ...]]]
}
```

Formato: [batch, seq_len, features] — ex: [1, 10, 16]

## Saída

Um vetor de 64 floats representando a codificação simbólica final.

## Exemplo

```python
import replicate

output = replicate.run(
    "seu-usuario/sage-two",
    input={"sequence": [[[0.1]*16]*10]}
)
```
