import threading
import time
from pathlib import Path

import streamlit as st
import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

PASTA = Path(__file__).resolve().parent
CAMINHO_MODELO = (
    PASTA / "resultados" / "experimento_07" / "modelo"
)

st.set_page_config(
    page_title="Assistente de Seguros",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def carregar_modelo():
    if not (CAMINHO_MODELO / "config.json").exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "distilgpt2",
        clean_up_tokenization_spaces=False,
    )

    device = torch_directml.device()

    model = AutoModelForCausalLM.from_pretrained(
        str(CAMINHO_MODELO),
        attn_implementation="eager",
    )

    model.to(device)
    model.tie_weights()
    model.config.use_cache = False
    model.eval()

    assert (
        model.get_input_embeddings().weight
        is model.get_output_embeddings().weight
    ), "Os pesos do modelo não estão compartilhados."

    return (
        tokenizer,
        model,
        device,
        torch_directml.device_name(0),
        threading.Lock(),
    )


st.title("Assistente de Seguros")
st.write(
    "Escreva uma solicitação em inglês para receber uma "
    "orientação sobre comunicação de incidentes, acompanhamento "
    "de sinistros ou renovação de apólices."
)

st.info(
    "Demonstração acadêmica com orientações para um portal fictício. "
    "O assistente não consulta apólices nem o status real de sinistros."
)

with st.expander("Exemplos de perguntas"):
    st.code(
        "How can I report an incident?\n"
        "Where can I check the status of my claim?\n"
        "How can I renew my insurance policy?",
        language=None,
    )

with st.form("formulario"):
    pergunta = st.text_area(
        "Sua solicitação em inglês",
        placeholder="How can I renew my insurance policy?",
        height=120,
        max_chars=1000,
    )
    enviar = st.form_submit_button("Gerar resposta")

if enviar:
    pergunta = pergunta.strip()

    if not pergunta:
        st.warning("Escreva uma solicitação antes de continuar.")
    else:
        try:
            with st.spinner("Gerando resposta na GPU..."):
                tokenizer, model, device, gpu, lock = carregar_modelo()

                prompt = f"Customer: {pergunta}\nAssistant:"
                entrada = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )

                if entrada["input_ids"].shape[1] > 96:
                    st.warning(
                        "A solicitação está muito longa. "
                        "Resuma a pergunta para continuar."
                    )
                else:
                    entrada = {
                        chave: valor.to(device)
                        for chave, valor in entrada.items()
                    }

                    inicio = time.perf_counter()

                    with lock, torch.no_grad():
                        saida = model.generate(
                            **entrada,
                            max_new_tokens=160,
                            do_sample=False,
                            use_cache=False,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                        novos_tokens = saida[
                            0, entrada["input_ids"].shape[1]:
                        ].cpu()

                    resposta = tokenizer.decode(
                        novos_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()

                    duracao = time.perf_counter() - inicio

                    st.session_state["ultima_resposta"] = {
                        "pergunta": pergunta,
                        "resposta": resposta,
                        "gpu": gpu,
                        "duracao": duracao,
                    }

        except Exception as erro:
            st.error("Não foi possível gerar a resposta.")
            st.exception(erro)

if "ultima_resposta" in st.session_state:
    resultado = st.session_state["ultima_resposta"]

    st.subheader("Resposta gerada")
    st.caption(f"Solicitação: {resultado['pergunta']}")
    st.write(
        resultado["resposta"]
        or "O modelo não gerou uma resposta."
    )

    st.caption(
        f"GPU: {resultado['gpu']} · "
        f"Tempo de geração: {resultado['duracao']:.2f} segundos"
    )

with st.expander("Sobre o projeto"):
    st.write(
        "DistilGPT2 ajustado localmente com 300 exemplos, "
        "utilizando GPU AMD via DirectML."
    )
    st.write(
        "No teste reservado, 56 de 60 respostas corresponderam "
        "exatamente às orientações esperadas (93,3%). "
        "Foram identificadas quatro falhas de conteúdo."
    )
    st.write(
        "Escopo: três intenções, com respostas padronizadas em inglês. "
        "Perguntas fora desse escopo podem produzir respostas inadequadas."
    )