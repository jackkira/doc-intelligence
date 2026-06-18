import streamlit as st
import openai
import pypdf
import docx
import io
import tiktoken

st.set_page_config(page_title="Document Intelligence", layout="wide")


def parse_document(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file.read()))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return file.read().decode("utf-8", errors="ignore")


def maybe_truncate(text: str, max_tokens: int = 100_000) -> str:
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        return enc.decode(tokens[:max_tokens])
    return text


def call_openai(prompt: str) -> str:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def extract_rules(text: str) -> str:
    prompt = (
        "Read the document below and extract every explicit rule, policy, constraint, "
        "or requirement stated in it. Output a numbered list only — one rule per line. "
        "Do not infer or add rules that are not directly stated in the document.\n\n"
        f"DOCUMENT:\n{text}"
    )
    return call_openai(prompt)


def answer_question(text: str, question: str) -> str:
    prompt = (
        "Answer the question using ONLY the information in the document below. "
        "If the answer is not present in the document, say: "
        "'I cannot find that information in the document.'\n\n"
        f"DOCUMENT:\n{text}\n\n"
        f"QUESTION: {question}"
    )
    return call_openai(prompt)


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("Document Intelligence")
st.caption("Upload a document to extract its rules and ask questions about it.")

uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if uploaded:
    if "doc_name" not in st.session_state or st.session_state.doc_name != uploaded.name:
        with st.spinner("Reading document and extracting rules..."):
            text = parse_document(uploaded)
            text = maybe_truncate(text)
            rules = extract_rules(text)
        st.session_state.doc_name = uploaded.name
        st.session_state.doc_text = text
        st.session_state.rules = rules
        st.session_state.qa_history = []

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Derived Rules")
        st.markdown(st.session_state.rules)

    with right:
        st.subheader("Ask a Question")
        question = st.text_input("Your question", key="question_input")
        if st.button("Ask") and question.strip():
            with st.spinner("Thinking..."):
                answer = answer_question(st.session_state.doc_text, question)
            st.session_state.qa_history.append((question, answer))

        for q, a in reversed(st.session_state.qa_history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.divider()
