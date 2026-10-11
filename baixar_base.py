from pathlib import Path
from datasets import load_dataset

PASTA = Path(__file__).resolve().parent / "dados"
PASTA.mkdir(exist_ok=True)

print("Baixando a base pública de seguros...")

base = load_dataset(
    "bitext/Bitext-insurance-llm-chatbot-training-dataset",
    split="train",
)

print("Total de exemplos:", len(base))
print("Colunas:", base.column_names)

base.to_json(
    str(PASTA / "seguros_original.jsonl"),
    force_ascii=False,
)

print("\nPrimeiro exemplo:")
print(base[0])
print("\nBase salva em:", PASTA)