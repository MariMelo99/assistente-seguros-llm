# Assistente de Seguros com LLM Local

Projeto acadêmico de CP2/CP3 — Generative AI for Engineering.

Ajuste fino do DistilGPT2 para gerar respostas em inglês sobre três solicitações de atendimento de seguros:

- Renovar uma apólice.
- Comunicar um incidente.
- Acompanhar um sinistro.

O treinamento e a geração das respostas foram executados localmente no Windows, utilizando GPU AMD com DirectML.

## Modelo escolhido

Foi utilizado o **DistilGPT2**, um modelo de linguagem generativo compacto, adequado para experimentar treinamento local com recursos limitados.

Modelo original: https://huggingface.co/distilgpt2

O projeto ajusta os pesos do modelo com exemplos de perguntas e respostas. A intenção presente no dataset é usada na preparação e análise dos dados; na interface, o usuário fornece apenas a pergunta.

## Ambiente

- Sistema operacional: Windows.
- GPU: AMD Radeon(TM) Graphics.
- Backend de aceleração: DirectML.
- Dispositivo apresentado pelo PyTorch: `privateuseone:0`.
- Python: 3.10.
- PyTorch e torch-directml.
- Transformers.
- Streamlit para o frontend.

CUDA é uma tecnologia da NVIDIA. Como a GPU utilizada é AMD, o projeto utiliza DirectML.

A tokenização e o salvamento dos arquivos utilizam CPU. As operações do modelo no treinamento, na validação e na geração são executadas no dispositivo DirectML.

## Origem e preparação dos dados

Dataset público da Bitext:

https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset

A base utilizada contém 39.000 exemplos distribuídos em 39 intenções.

Foram selecionadas três intenções:

| Intenção | Solicitação |
|---|---|
| `renew_insurance_policy` | Renovação de apólice |
| `report_incident` | Comunicação de incidente |
| `track_claim` | Acompanhamento de sinistro |

A preparação incluiu:

- Remoção de perguntas com palavrões.
- Remoção de perguntas duplicadas.
- Agrupamento de perguntas por similaridade lexical antes da divisão.
- Seleção de exemplos balanceados por intenção.
- Adaptação das respostas para procedimentos curtos de um portal fictício.

Após a limpeza, restaram 1.516 exemplos. Para os experimentos foram selecionados:

| Conjunto | Exemplos | Exemplos por intenção |
|---|---:|---:|
| Treino | 300 | 100 |
| Validação | 60 | 20 |
| Teste | 60 | 20 |

Cada intenção possui uma resposta curta padronizada. Assim, o objetivo é gerar o procedimento correspondente à pergunta.

Os dados têm origem na Bitext. Consulte a página original para a licença e as condições de utilização.

## Treinamento

Configuração dos experimentos finais:

- Otimizador: SGD.
- Batch size: 1.
- Épocas: 2.
- Passos: 600.
- Semente: 42.
- Limite do prompt: 96 tokens.
- Limite da resposta: 160 tokens, incluindo o token de fim.
- Cálculo da loss somente nos tokens da resposta.

O compartilhamento dos pesos de entrada e saída do DistilGPT2 é restabelecido com `model.tie_weights()` após a transferência de dispositivo.

## Comparação de parâmetros

Os experimentos 07 e 08 utilizaram os mesmos conjuntos e respostas curtas.

| Experimento | Learning rate | Épocas | Loss de validação final | Duração aproximada |
|---|---:|---:|---:|---:|
| 07 | 0.001 | 2 | 0.01747 | 2,27 minutos |
| 08 | 0.0005 | 2 | 0.06980 | 2,28 minutos |

A loss inicial de validação foi aproximadamente 8.80108.

O experimento 07 foi selecionado pela menor loss de validação.

Experimentos anteriores com respostas longas apresentaram repetição e procedimentos incorretos. A adaptação para respostas curtas melhorou os resultados dentro do escopo definido.

## Avaliação no conjunto de teste

A avaliação comparou o modelo original com o experimento 07 em 60 perguntas de teste.

A métrica foi correspondência exata com a resposta de referência, após normalização dos espaços.

| Modelo | Correspondências exatas | Taxa |
|---|---:|---:|
| DistilGPT2 original | 0/60 | 0% |
| Experimento 07 | 56/60 | 93,33% |

Resultados do experimento 07 por intenção:

| Intenção | Correspondências exatas |
|---|---:|
| Renovação de apólice | 20/20 |
| Comunicação de incidente | 19/20 |
| Acompanhamento de sinistro | 17/20 |

Os quatro casos diferentes da referência apresentaram confusão entre procedimentos ou mistura de instruções.

O resultado de 93,33% se refere exclusivamente a esse conjunto de teste e às três respostas padronizadas. Não representa precisão geral em assuntos de seguros.

## Como executar no Windows

Abra o PowerShell na pasta do projeto.

### 1. Criar o ambiente e instalar as dependências

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Preparar os dados

```powershell
.\.venv\Scripts\python.exe baixar_base.py
.\.venv\Scripts\python.exe preparar_base.py
.\.venv\Scripts\python.exe separar_base.py
.\.venv\Scripts\python.exe adaptar_respostas.py
```

### 3. Treinar os modelos

```powershell
.\.venv\Scripts\python.exe treinar.py --lr 0.001 --epocas 2 --nome experimento_07
.\.venv\Scripts\python.exe treinar.py --lr 0.0005 --epocas 2 --nome experimento_08
```

O treinamento salva os pesos localmente em `resultados/<experimento>/modelo`.

Os pesos não estão incluídos no GitHub. É necessário executar o treinamento para utilizá-los.

O script impede sobrescrever uma pasta de experimento existente. Se ela já contiver as métricas publicadas, faça uma cópia dessas métricas fora da pasta antes de remover a pasta correspondente e reproduzir o treinamento.

### 4. Avaliar

```powershell
.\.venv\Scripts\python.exe comparar_respostas.py
.\.venv\Scripts\python.exe avaliar_teste.py
.\.venv\Scripts\python.exe ver_erros.py
```

### 5. Abrir o frontend

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Acesse o endereço local informado no terminal, normalmente `http://localhost:8501`.

## Exemplos de prompts

Digite as perguntas em inglês:

- `How can I renew my insurance policy?`
- `How do I report an incident?`
- `How can I track my claim?`

## Evidências

Os prints da GPU funcionando no VSCode e da interface estão na pasta `docs`.

As métricas e respostas geradas estão na pasta `resultados`.

## Limitações

- O modelo foi ajustado para apenas três intenções.
- As perguntas e respostas são em inglês.
- As respostas representam um portal fictício.
- O sistema não consulta apólices ou sinistros reais.
- Perguntas fora do escopo podem gerar respostas incorretas.
- O conjunto de teste é pequeno e contém variações de uma base sintética.
- O agrupamento lexical reduz similaridades entre conjuntos, mas não garante separação semântica completa.
- A baixa loss também reflete a padronização das três respostas.

## Integrantes
Generative AI for Engineering
Turma: 4ESPV 
Nome: Irana Pereira – RM 98593 
Nome: Lucas Vinicius Candido da Silva – RM 98480 
Nome: Mariana Melo – RM 98121 
