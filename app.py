import threading
import time
from pathlib import Path

import streamlit as st
import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

PASTA = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA / "resultados" / "experimento_07" / "modelo"

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


def limpar():
    st.session_state["pergunta"] = ""
    st.session_state.pop("ultima_resposta", None)


st.title("Assistente de Seguros")

st.write(
    "Escreva uma solicitação em inglês sobre comunicação "
    "de incidentes, acompanhamento de sinistros ou "
    "renovação de apólices."
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

st.text_area(
    "Sua solicitação em inglês",
    key="pergunta",
    placeholder="How can I renew my insurance policy?",
    height=120,
    max_chars=1000,
)

coluna_enviar, coluna_limpar = st.columns(2)

with coluna_enviar:
    enviar = st.button(
        "Gerar resposta",
        type="primary",
        use_container_width=True,
    )

with coluna_limpar:
    st.button(
        "Limpar",
        on_click=limpar,
        use_container_width=True,
    )

if enviar:
    pergunta = st.session_state["pergunta"].strip()

    # Remove a resposta anterior, inclusive quando há erro no novo envio.
    st.session_state.pop("ultima_resposta", None)

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

                tamanho_prompt = entrada["input_ids"].shape[1]

                if tamanho_prompt > 96:
                    st.warning(
                        "A solicitação está muito longa. "
                        "Resuma a pergunta para continuar."
                    )
                else:
                    inicio = time.perf_counter()

                    with lock, torch.no_grad():
                        entrada_gpu = {
                            chave: valor.to(device)
                            for chave, valor in entrada.items()
                        }

                        saida = model.generate(
                            **entrada_gpu,
                            max_new_tokens=160,
                            do_sample=False,
                            use_cache=False,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                        novos_tokens = saida[0, tamanho_prompt:].cpu()

                    resposta = tokenizer.decode(
                        novos_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()

                    st.session_state["ultima_resposta"] = {
                        "pergunta": pergunta,
                        "resposta": resposta,
                        "gpu": gpu,
                        "duracao": time.perf_counter() - inicio,
                    }

        except Exception as erro:
            st.error("Não foi possível gerar a resposta.")
            st.exception(erro)

resultado = st.session_state.get("ultima_resposta")

if resultado:
    st.subheader("Resposta gerada")
    st.caption(f"Solicitação: {resultado['pergunta']}")
    st.write(
        resultado["resposta"] or "O modelo não gerou uma resposta."
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