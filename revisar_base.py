import json
import random
from pathlib import Path

pasta = Path(__file__).resolve().parent / "dados"
intencoes = [
    "report_incident",
    "track_claim",
    "renew_insurance_policy",
]

grupos = {nome: [] for nome in intencoes}

with (pasta / "seguros_original.jsonl").open(encoding="utf-8") as f:
    for linha in f:
        item = json.loads(linha)
        if item["intent"] in grupos:
            grupos[item["intent"]].append(item)

rng = random.Random(42)
selecionados = []

for intencao, exemplos in grupos.items():
    selecionados.extend(rng.sample(exemplos, min(3, len(exemplos))))

destino = pasta / "amostra_revisao.txt"

with destino.open("w", encoding="utf-8") as f:
    for item in selecionados:
        texto = (
            f"INTENÇÃO: {item['intent']}\n"
            f"PERGUNTA: {item['instruction']}\n"
            f"RESPOSTA: {item['response']}\n"
            + "=" * 60 + "\n\n"
        )
        f.write(texto)
        print(texto)

print("Revisão salva em:", destino)