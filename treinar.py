import argparse
import json
import time
from pathlib import Path

import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epocas", type=int, default=1)
parser.add_argument("--passos", type=int, default=0)
parser.add_argument("--nome", default="experimento_04")
args = parser.parse_args()

if args.lr <= 0:
    parser.error("--lr precisa ser maior que zero.")

if args.epocas < 1:
    parser.error("--epocas precisa ser pelo menos 1.")

if args.passos < 0:
    parser.error("--passos não pode ser negativo.")

torch.manual_seed(42)

pasta = Path(__file__).resolve().parent
destino = pasta / "resultados" / args.nome

if destino.exists():
    raise FileExistsError(
        f"{destino} já existe. Use outro valor para --nome."
    )


def ler(nome):
    arquivo = pasta / "dados" / f"{nome}.jsonl"

    with arquivo.open(encoding="utf-8") as f:
        itens = [
            json.loads(linha)
            for linha in f
            if linha.strip()
        ]

    if not itens:
        raise ValueError(f"O arquivo {arquivo} está vazio.")

    return itens


treino = ler("treino_curto")
validacao = ler("validacao_curto")

device = torch_directml.device()

print("GPU:", torch_directml.device_name(0), flush=True)
print("Dispositivo:", device, flush=True)
print("Carregando DistilGPT2...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    "distilgpt2",
    clean_up_tokenization_spaces=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    attn_implementation="eager",
)

model.config.use_cache = False
model.to(device)
model.tie_weights()

assert (
    model.get_input_embeddings().weight
    is model.get_output_embeddings().weight
), "Os pesos precisam estar compartilhados antes do treinamento."


def preparar(item):
    prompt = f"Customer: {item['instruction']}\nAssistant:"

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    resposta_ids = tokenizer.encode(
        " " + item["response"],
        add_special_tokens=False,
    )

    # Até 96 tokens de entrada e 160 de resposta.
    # O último token da resposta sempre indica seu fim.
    prompt_ids = prompt_ids[:96]
    resposta_ids = resposta_ids[:159] + [
        tokenizer.eos_token_id
    ]

    ids = prompt_ids + resposta_ids

    # Ignora o prompt no cálculo do erro.
    labels = [-100] * len(prompt_ids) + resposta_ids

    return {
        "input_ids": torch.tensor(
            [ids],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.ones(
            (1, len(ids)),
            dtype=torch.long,
            device=device,
        ),
        "labels": torch.tensor(
            [labels],
            dtype=torch.long,
            device=device,
        ),
    }


def avaliar(itens):
    model.eval()
    soma = 0.0
    total_tokens = 0

    with torch.no_grad():
        for item in itens:
            batch = preparar(item)
            loss = model(**batch).loss

            if not torch.isfinite(loss).cpu().item():
                raise RuntimeError(
                    "Erro não finito durante a avaliação."
                )

            tokens = (
                batch["labels"][:, 1:] != -100
            ).sum().cpu().item()

            soma += loss.cpu().item() * tokens
            total_tokens += tokens

    return soma / total_tokens


def contar_truncamentos(itens):
    prompts_longos = 0
    respostas_longas = 0

    for item in itens:
        prompt = f"Customer: {item['instruction']}\nAssistant:"

        if len(tokenizer.encode(
            prompt, add_special_tokens=False
        )) > 96:
            prompts_longos += 1

        if len(tokenizer.encode(
            " " + item["response"],
            add_special_tokens=False,
        )) > 159:
            respostas_longas += 1

    return {
        "prompts_truncados": prompts_longos,
        "respostas_truncadas": respostas_longas,
    }


# No teste rápido, usa dois exemplos por intenção na validação.
if args.passos:
    contagem = {}
    amostra = []

    for item in validacao:
        intencao = item["intent"]

        if contagem.get(intencao, 0) < 2:
            amostra.append(item)
            contagem[intencao] = (
                contagem.get(intencao, 0) + 1
            )

    validacao = amostra


truncamentos = {
    "treino": contar_truncamentos(treino),
    "validacao": contar_truncamentos(validacao),
}

print("Exemplos de treino:", len(treino), flush=True)
print("Exemplos de validação:", len(validacao), flush=True)
print("Truncamentos:", truncamentos, flush=True)

inicio = time.time()

print("Avaliando modelo original...", flush=True)
antes = avaliar(validacao)
print("Erro de validação antes:", antes, flush=True)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
)

historico = []
passo = 0

for epoca in range(args.epocas):
    model.train()
    ordem = torch.randperm(len(treino)).tolist()

    for indice in ordem:
        batch = preparar(treino[indice])

        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss

        if not torch.isfinite(loss.detach()).cpu().item():
            raise RuntimeError(
                "Erro não finito; treinamento interrompido."
            )

        loss.backward()
        optimizer.step()

        passo += 1
        valor = loss.detach().cpu().item()

        historico.append({
            "passo": passo,
            "epoca": epoca + 1,
            "loss": valor,
        })

        if passo == 1 or passo % 10 == 0 or args.passos:
            print(
                f"Época {epoca + 1} | "
                f"passo {passo} | erro {valor:.4f}",
                flush=True,
            )

        if args.passos and passo >= args.passos:
            break

    if args.passos and passo >= args.passos:
        break


optimizer.zero_grad(set_to_none=True)
depois = avaliar(validacao)
duracao = (time.time() - inicio) / 60

print("Erro de validação depois:", depois, flush=True)
print(
    "Dispositivo dos pesos:",
    next(model.parameters()).device,
    flush=True,
)
print("Duração em minutos:", round(duracao, 2), flush=True)

destino.mkdir(parents=True)

registro = {
    "modelo": "distilgpt2",
    "gpu": torch_directml.device_name(0),
    "backend": "DirectML",
    "seed": 42,
    "otimizador": "SGD",
    "lr": args.lr,
    "epocas_solicitadas": args.epocas,
    "passos_executados": passo,
    "teste_rapido": bool(args.passos),
    "exemplos_treino": len(treino),
    "exemplos_validacao": len(validacao),
    "max_tokens_prompt": 96,
    "max_tokens_resposta": 160,
    "eos_preservado": True,
    "truncamentos": truncamentos,
    "duracao_minutos": duracao,
    "loss_validacao_antes": antes,
    "loss_validacao_depois": depois,
    "historico": historico,
}

with (destino / "metricas.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(
        registro,
        f,
        ensure_ascii=False,
        indent=2,
    )

# Transfere para CPU somente para salvar os arquivos.
if not args.passos:
    model.to("cpu")
model.tie_weights()
model.save_pretrained(destino / "modelo")

print("Resultados salvos em:", destino)