import streamlit as st

from assistant import AppleWatchAssistant
from config import LOCAL_LLM_MODEL, MAX_HISTORY_MESSAGES


st.set_page_config(page_title="Apple Watch Assistant", page_icon="⌚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_assistant(provider: str):
    return AppleWatchAssistant(provider)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("⌚ Apple Watch Assistant")
    provider_label = st.radio(
        "AI Provider",
        ["Groq", "Offline Hugging Face"],
        index=0,
        help="Choose the response model. Hugging Face runs locally after its model is downloaded.",
    )
    provider = "groq" if provider_label == "Groq" else "huggingface"

    if provider == "huggingface":
        st.caption(f"Local model: {LOCAL_LLM_MODEL}")
    else:
        st.caption("Cloud model: Groq")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

assistant = get_assistant(provider)

st.title("Apple Watch Assistant")
st.caption("Apple Watch assistance powered by static guides and live Apple.com information")

try:
    status = assistant.status()
    p = status["provider"]
    if p.get("available", True) is False:
        st.sidebar.error(f"{provider_label} not configured")
        if p.get("error"):
            st.sidebar.caption(p["error"])
    else:
        st.sidebar.success(f"{p.get('provider', provider_label)} ready")
    st.sidebar.caption(f"PDF chunks: {status['retrieval']['documents']}")
    st.sidebar.caption(f"Apple.com reachable: {'Yes' if status['apple_reachable'] else 'No'}")
except Exception as exc:
    st.sidebar.error(f"Initialization error: {exc}")

if not st.session_state.messages:
    st.info(
        "Ask me about Apple Watch setup, troubleshooting, features, comparisons, recommendations, "
        "or current Apple pricing and availability."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(source)

prompt = st.chat_input("Ask anything about Apple Watch...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1][-MAX_HISTORY_MESSAGES:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = assistant.answer(prompt, history)
                response = result["response"]
                sources = result["web_sources"] + [
                    f"PDF: {x.get('source')} (page {x.get('page', '?')})"
                    for x in result["pdf_sources"]
                ]
                st.markdown(response)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.caption(source)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                })
            except Exception as exc:
                st.error(f"I couldn't generate a response: {exc}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't generate a response with the selected AI provider. Please check its setup and try again.",
                })
