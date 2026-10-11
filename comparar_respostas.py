import gc
import json
from pathlib import Path

import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

PASTA = Path(__file__).resolve().parent
CAMINHO_AJUSTADO = (
    PASTA / "resultados" / "experimento_07" / "modelo"
)
ARQUIVO_VALIDACAO = PASTA / "dados" / "validacao_curto.jsonl"

if not (CAMINHO_AJUSTADO / "config.json").exists():
    raise FileNotFoundError(
        f"Modelo não encontrado em: {CAMINHO_AJUSTADO}"
    )

device = torch_directml.device()

tokenizer = AutoTokenizer.from_pretrained(
    "distilgpt2",
    clean_up_tokenization_spaces=False,
)

perguntas = {}

with ARQUIVO_VALIDACAO.open(encoding="utf-8") as f:
    for linha in f:
        if linha.strip():
            item = json.loads(linha)
            perguntas.setdefault(item["intent"], item)

resultados = []

modelos = {
    "original": "distilgpt2",
    "ajustado": str(CAMINHO_AJUSTADO),
}

print("GPU:", torch_directml.device_name(0), flush=True)
print("Base de validação:", ARQUIVO_VALIDACAO, flush=True)

for nome, caminho in modelos.items():
    print(f"\nCarregando modelo {nome}...", flush=True)
    print("CAMINHO DO MODELO:", caminho, flush=True)

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
    ), "Os pesos não estão compartilhados."

    for intencao, item in sorted(perguntas.items()):
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

        tamanho_prompt = entrada["input_ids"].shape[1]
        novos_tokens = saida[0, tamanho_prompt:].cpu()

        resposta = tokenizer.decode(
            novos_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        resultados.append({
            "modelo": nome,
            "caminho_modelo": caminho,
            "intent": intencao,
            "pergunta": item["instruction"],
            "resposta": resposta,
            "referencia": item["response"],
            "geracao": {
                "max_new_tokens": 160,
                "do_sample": False,
                "use_cache": False,
            },
        })

        print(f"\nMODELO: {nome}")
        print("INTENÇÃO:", intencao)
        print("PERGUNTA:", item["instruction"])
        print("RESPOSTA:", resposta, flush=True)

        del entrada, saida, novos_tokens

    del model
    gc.collect()

destino = (
    PASTA / "resultados" / "comparacao_validacao_exp07_conferida.json"
)
destino.parent.mkdir(parents=True, exist_ok=True)

with destino.open("w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\nComparação salva em:", destino)