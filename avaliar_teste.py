import gc
import json
from collections import Counter
from pathlib import Path

import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

PASTA = Path(__file__).resolve().parent
device = torch_directml.device()

with (PASTA / "dados" / "teste_curto.jsonl").open(
    encoding="utf-8"
) as f:
    teste = [json.loads(linha) for linha in f if linha.strip()]

tokenizer = AutoTokenizer.from_pretrained(
    "distilgpt2",
    clean_up_tokenization_spaces=False,
)

modelos = {
    "original": "distilgpt2",
    "experimento_07": str(
        PASTA / "resultados" / "experimento_07" / "modelo"
    ),
}

resumos = {}
respostas = []


def normalizar(texto):
    return " ".join(texto.split())


print("GPU:", torch_directml.device_name(0), flush=True)
print("Perguntas de teste:", len(teste), flush=True)

for nome, caminho in modelos.items():
    print(f"\nAvaliando {nome}...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        caminho,
        attn_implementation="eager",
    )
    model.to(device)
    model.tie_weights()
    model.config.use_cache = False
    model.eval()

    assert (
        model.get_input_embeddings().weight
        is model.get_output_embeddings().weight
    )

    acertos = Counter()
    totais = Counter()

    for numero, item in enumerate(teste, start=1):
        prompt = f"Customer: {item['instruction']}\nAssistant:"

        entrada = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=96,
        )
        entrada = {
            chave: valor.to(device)
            for chave, valor in entrada.items()
        }

        with torch.no_grad():
            saida = model.generate(
                **entrada,
                max_new_tokens=160,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        novos = saida[0, entrada["input_ids"].shape[1]:].cpu()
        resposta = tokenizer.decode(
            novos,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        correto = (
            normalizar(resposta) == normalizar(item["response"])
        )

        intencao = item["intent"]
        totais[intencao] += 1
        acertos[intencao] += int(correto)

        respostas.append({
            "modelo": nome,
            "intent": intencao,
            "pergunta": item["instruction"],
            "referencia": item["response"],
            "resposta": resposta,
            "correspondencia_exata": correto,
        })

        print(
            f"{nome}: {numero}/{len(teste)} | "
            f"correspondência exata: {correto}",
            flush=True,
        )

        del entrada, saida, novos

    total_acertos = sum(acertos.values())

    resumo = {
        "acertos": total_acertos,
        "total": len(teste),
        "taxa_correspondencia_exata": total_acertos / len(teste),
        "por_intencao": {
            intencao: {
                "acertos": acertos[intencao],
                "total": totais[intencao],
            }
            for intencao in sorted(totais)
        },
    }
    resumos[nome] = resumo

    print(f"\nRESULTADO: {nome}")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))

    del model
    gc.collect()

destino = PASTA / "resultados" / "avaliacao_teste.json"

with destino.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "gpu": torch_directml.device_name(0),
            "backend": "DirectML",
            "criterio": "Correspondência exata após normalizar espaços",
            "resumos": resumos,
            "respostas": respostas,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nAvaliação salva em:", destino)
