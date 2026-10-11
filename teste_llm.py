import torch
import torch_directml
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)
device = torch_directml.device()

print("GPU:", torch_directml.device_name(0))
print("Carregando modelo...")

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    attn_implementation="eager",
)
model.config.use_cache = False
model.to(device)
model.train()

text = (
    "Customer: How can I check my insurance policy? "
    "Assistant: You can review your policy documents."
)

batch = tokenizer(text, return_tensors="pt")
batch = {key: value.to(device) for key, value in batch.items()}
labels = batch["input_ids"].clone()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

print("Executando um passo de treinamento...")
optimizer.zero_grad()

output = model(**batch, labels=labels)
print("Erro antes:", output.loss.detach().cpu().item())

output.loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

with torch.no_grad():
    after = model(**batch, labels=labels).loss

print("Erro depois:", after.cpu().item())
print("Dispositivo dos pesos:", next(model.parameters()).device)
print("Teste concluído.")