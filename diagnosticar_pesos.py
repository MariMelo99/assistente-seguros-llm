import torch_directml
from transformers import AutoModelForCausalLM

device = torch_directml.device()

model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    attn_implementation="eager",
)


def verificar(etapa):
    entrada = model.get_input_embeddings().weight
    saida = model.get_output_embeddings().weight

    print(f"\n{etapa}")
    print("Pesos são o mesmo objeto:", entrada is saida)
    print("Dispositivo da entrada:", entrada.device)
    print("Dispositivo da saída:", saida.device)


print("GPU:", torch_directml.device_name(0))

verificar("Modelo original na CPU")

model.to(device)
verificar("Após transferir para DirectML")

model.tie_weights()
verificar("Após restabelecer o compartilhamento")