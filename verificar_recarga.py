import json
from pathlib import Path

import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

pasta = Path(__file__).resolve().parent
experimento = pasta / "resultados" / "experimento_07"
device = torch_directml.device()

tokenizer = AutoTokenizer.from_pretrained(
    "distilgpt2",
    clean_up_tokenization_spaces=False,
)

model = AutoModelForCausalLM.from_pretrained(
    experimento / "modelo",
    attn_implementation="eager",
)
model.to(device)
model.tie_weights()
model.config.use_cache = False
model.eval()

with (pasta / "dados" / "validacao_curto.jsonl").open(
    encoding="utf-8"
) as f:
    itens = [json.loads(linha) for linha in f if linha.strip()]

soma = 0.0
total_tokens = 0

with torch.no_grad():
    for item in itens:
        prompt = f"Customer: {item['instruction']}\nAssistant:"
        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False
        )[:96]
        resposta_ids = tokenizer.encode(
            " " + item["response"],
            add_special_tokens=False,
        )[:159] + [tokenizer.eos_token_id]

        ids = prompt_ids + resposta_ids
        labels = [-100] * len(prompt_ids) + resposta_ids

        loss = model(
            input_ids=torch.tensor([ids], device=device),
            attention_mask=torch.ones(
                (1, len(ids)),
                dtype=torch.long,
                device=device,
            ),
            labels=torch.tensor([labels], device=device),
        ).loss.cpu().item()

        soma += loss * len(resposta_ids)
        total_tokens += len(resposta_ids)

erro_recarga = soma / total_tokens

with (experimento / "metricas.json").open(
    encoding="utf-8"
) as f:
    metricas = json.load(f)

erro_salvo = metricas["loss_validacao_depois"]

print("GPU:", torch_directml.device_name(0))
print("Erro ao terminar o treinamento:", erro_salvo)
print("Erro após recarregar:", erro_recarga)
print("Diferença:", abs(erro_salvo - erro_recarga))
print(
    "Pesos compartilhados:",
    model.get_input_embeddings().weight
    is model.get_output_embeddings().weight,
)