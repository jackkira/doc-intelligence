import io
import numpy as np
import faiss
import openai
import pypdf
import docx
import streamlit as st

st.set_page_config(page_title="Document Intelligence", layout="wide")


def parse_document(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file.read()))
        return "\n".join(p.text for p in doc.paragraphs)
    return file.read().decode("utf-8", errors="ignore")


def get_client():
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def embed(texts: list[str]) -> np.ndarray:
    client = get_client()
    all_vecs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        all_vecs.extend([r.embedding for r in resp.data])
    vecs = np.array(all_vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def build_index(text: str):
    size, overlap = 1000, 200
    chunks = [text[i:i+size] for i in range(0, len(text), size - overlap)]
    vecs = embed(chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, chunks


def retrieve(question: str, index, chunks: list[str], k: int = 5) -> str:
    vec = embed([question])
    _, ids = index.search(vec, k)
    return "\n\n".join(chunks[i] for i in ids[0] if i < len(chunks))


def call_llm(prompt: str) -> str:
    resp = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def extract_rules(text: str) -> str:
    return call_llm(
        "Extract every explicit rule, policy, constraint, or requirement from the document below. "
        "Output a numbered list only — one rule per line. Do not infer rules not stated in the document.\n\n"
        f"DOCUMENT:\n{text}"
    )


def answer_question(question: str, index, chunks: list[str]) -> str:
    context = retrieve(question, index, chunks)
    return call_llm(
        "You are an assistant answering questions about a document. "
        "Use the document excerpts below to answer the question. "
        "If the answer cannot be found in the excerpts, say so.\n\n"
        f"DOCUMENT EXCERPTS:\n{context}\n\nQUESTION: {question}"
    )


# ── UI ───────────────────────────────────────────────────────────────────────

st.title("Document Intelligence")
st.caption("Upload a document to extract its rules and ask questions about it.")

uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if uploaded:
    if st.session_state.get("doc_name") != uploaded.name:
        with st.spinner("Processing document..."):
            text = parse_document(uploaded)
            index, chunks = build_index(text)
            rules = extract_rules(text)
        st.session_state.update(doc_name=uploaded.name, index=index, chunks=chunks, rules=rules, qa_history=[])

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Derived Rules")
        st.markdown(st.session_state.rules)

    with right:
        st.subheader("Ask a Question")
        question = st.text_input("Your question", key="q")
        if st.button("Ask") and question.strip():
            with st.spinner("Thinking..."):
                answer = answer_question(question, st.session_state.index, st.session_state.chunks)
            st.session_state.qa_history.append((question, answer))

        for q, a in reversed(st.session_state.qa_history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.divider()
