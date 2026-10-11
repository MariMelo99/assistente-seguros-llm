import json
from pathlib import Path

PASTA = Path(__file__).resolve().parent / "dados"

RESPOSTAS = {
    "report_incident": (
        "To report an incident, open 'Report an incident' "
        "on the insurer's portal, enter the incident details, "
        "review them and submit the form."
    ),
    "track_claim": (
        "To track a claim, sign in to the insurer's portal, "
        "open 'Claims', select your claim and review its status."
    ),
    "renew_insurance_policy": (
        "To renew a policy, sign in to the insurer's portal, "
        "open 'My policies', select the policy and review "
        "the renewal options."
    ),
}

for conjunto in ["treino", "validacao", "teste"]:
    origem = PASTA / f"{conjunto}.jsonl"
    destino = PASTA / f"{conjunto}_curto.jsonl"

    if destino.exists():
        raise FileExistsError(f"O arquivo já existe: {destino}")

    quantidade = 0

    with origem.open(encoding="utf-8") as entrada:
        with destino.open("w", encoding="utf-8") as saida:
            for linha in entrada:
                if not linha.strip():
                    continue

                item = json.loads(linha)

                adaptado = {
                    "instruction": item["instruction"],
                    "intent": item["intent"],
                    "response": RESPOSTAS[item["intent"]],
                    "response_source": (
                        "Resposta adaptada para um portal fictício."
                    ),
                }

                saida.write(
                    json.dumps(adaptado, ensure_ascii=False) + "\n"
                )
                quantidade += 1

    print(f"{conjunto}: {quantidade} exemplos — {destino}")

print("Arquivos originais preservados.")