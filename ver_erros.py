import json
from pathlib import Path

arquivo = (
    Path(__file__).resolve().parent
    / "resultados"
    / "avaliacao_teste.json"
)

with arquivo.open(encoding="utf-8") as f:
    dados = json.load(f)

erros = [
    item
    for item in dados["respostas"]
    if item["modelo"] == "experimento_07"
    and not item["correspondencia_exata"]
]

print("Respostas diferentes da referência:", len(erros))

for numero, item in enumerate(erros, start=1):
    print(f"\n--- CASO {numero} ---")
    print("INTENÇÃO:", item["intent"])
    print("PERGUNTA:", item["pergunta"])
    print("RESPOSTA GERADA:", item["resposta"])
    print("REFERÊNCIA:", item["referencia"])