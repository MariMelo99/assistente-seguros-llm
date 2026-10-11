import json
from collections import Counter
from pathlib import Path

arquivo = (
    Path(__file__).resolve().parent
    / "dados"
    / "seguros_original.jsonl"
)

contagem = Counter()

with arquivo.open(encoding="utf-8") as f:
    for linha in f:
        exemplo = json.loads(linha)
        contagem[exemplo["intent"]] += 1

print("EXEMPLOS POR INTENÇÃO\n")

for intencao, quantidade in sorted(contagem.items()):
    print(f"{intencao}: {quantidade}")

print(f"\nTotal de intenções: {len(contagem)}")
print(f"Total de exemplos: {sum(contagem.values())}")