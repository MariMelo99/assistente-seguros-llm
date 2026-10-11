import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PASTA = Path(__file__).resolve().parent / "dados"
SEED = 42
LIMITES = {"treino": 100, "validacao": 20, "teste": 20}

grupos_intencao = defaultdict(list)

with (PASTA / "seguros_preparados.jsonl").open(encoding="utf-8") as f:
    for linha in f:
        item = json.loads(linha)
        grupos_intencao[item["intent"]].append(item)

conjuntos = {nome: [] for nome in LIMITES}
rng = random.Random(SEED)

for intencao, itens in sorted(grupos_intencao.items()):
    palavras = [
        set(re.findall(r"\w+", item["instruction"].lower()))
        for item in itens
    ]
    pais = list(range(len(itens)))

    def raiz(i):
        while pais[i] != i:
            pais[i] = pais[pais[i]]
            i = pais[i]
        return i

    # Agrupa perguntas com alta sobreposição de palavras.
    for i in range(len(itens)):
        for j in range(i):
            uniao = palavras[i] | palavras[j]
            similaridade = (
                len(palavras[i] & palavras[j]) / len(uniao)
                if uniao else 1.0
            )
            if similaridade >= 0.80:
                pais[raiz(i)] = raiz(j)

    familias = defaultdict(list)
    for i, item in enumerate(itens):
        familias[raiz(i)].append(item)

    familias = list(familias.values())
    rng.shuffle(familias)

    if len(familias) < 3:
        raise ValueError(
            f"{intencao}: poucos grupos independentes para separar."
        )

    # Divide por grupos, nunca por exemplos de um mesmo grupo.
    n = len(familias)
    n_teste = max(1, round(n * 0.10))
    n_validacao = max(1, round(n * 0.10))

    divisao = {
        "teste": familias[:n_teste],
        "validacao": familias[n_teste:n_teste + n_validacao],
        "treino": familias[n_teste + n_validacao:],
    }

    for nome, familias_split in divisao.items():
        candidatos = [
            item
            for familia in familias_split
            for item in familia
        ]
        rng.shuffle(candidatos)
        conjuntos[nome].extend(candidatos[:LIMITES[nome]])

    print(f"{intencao}: {n} grupos de perguntas")

for nome, exemplos in conjuntos.items():
    rng.shuffle(exemplos)
    arquivo = PASTA / f"{nome}.jsonl"

    with arquivo.open("w", encoding="utf-8") as f:
        for item in exemplos:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{nome.upper()}: {len(exemplos)} exemplos")
    for intencao, quantidade in sorted(
        Counter(item["intent"] for item in exemplos).items()
    ):
        print(f"  {intencao}: {quantidade}")

    print("Salvo em:", arquivo)