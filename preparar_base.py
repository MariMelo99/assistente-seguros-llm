import json
import re
from collections import Counter
from pathlib import Path

PASTA = Path(__file__).resolve().parent / "dados"

INTENCOES = {
    "report_incident",
    "track_claim",
    "renew_insurance_policy",
}

MARCADORES = {
    "{{WEBSITE_URL}}": "the insurer's official website",
    "{{REPORT_INCIDENT_SECTION}}": "'Report an incident'",
    "{{SUBMIT_BUTTON}}": "'Submit'",
    "{{CLAIM_SECTION}}": "'Claims'",
    "{{INSURANCE_POLICY_SECTION}}": "'My policies'",
    "{{RENEW_INSURANCE_POLICY_OPTION}}": "'Renew policy'",
}

# Filtro inicial; não cobre todas as variações possíveis.
PALAVROES = re.compile(
    r"\b(fuck\w*|shit\w*|bullshit|bitch\w*|asshole\w*|"
    r"fucking|fuciing)\b",
    re.IGNORECASE,
)


def normalizar(texto):
    return " ".join(re.findall(r"\w+", texto.lower()))


contagem = Counter()
removidos = Counter()
vistos = set()
preparados = []

with (PASTA / "seguros_original.jsonl").open(encoding="utf-8") as f:
    for linha in f:
        item = json.loads(linha)

        if item["intent"] not in INTENCOES:
            continue

        pergunta = item["instruction"].strip()
        resposta = item["response"].strip()

        if not pergunta or not resposta:
            removidos["texto vazio"] += 1
            continue

        if PALAVROES.search(pergunta):
            removidos["palavrão"] += 1
            continue

        for marcador, substituto in MARCADORES.items():
            resposta = resposta.replace(marcador, substituto)

        # Não inventar substitutos para marcadores desconhecidos.
        if "{{" in resposta or "}}" in resposta:
            removidos["marcador desconhecido"] += 1
            continue

        chave = normalizar(pergunta)
        if chave in vistos:
            removidos["pergunta duplicada"] += 1
            continue

        vistos.add(chave)
        preparados.append({
            "instruction": pergunta,
            "response": resposta,
            "intent": item["intent"],
        })
        contagem[item["intent"]] += 1

destino = PASTA / "seguros_preparados.jsonl"

with destino.open("w", encoding="utf-8") as f:
    for item in preparados:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("EXEMPLOS PREPARADOS:")
for intencao, quantidade in sorted(contagem.items()):
    print(f"{intencao}: {quantidade}")

print("\nEXEMPLOS REMOVIDOS:")
for motivo, quantidade in sorted(removidos.items()):
    print(f"{motivo}: {quantidade}")

print("\nTotal preparado:", len(preparados))
print("Arquivo salvo:", destino)