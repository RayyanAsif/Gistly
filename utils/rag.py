import os
import uuid
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

from utils.chains import get_llm
from utils.transcribe import format_timestamp

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

def build_document(segment, texts: list[str], max_chars: int = 800) -> list[Document]:
    """Group consecutive segements into max char chunks, keeping start/end times"""
    if len(segment) != len(texts):
        raise ValueError(
            f"Transcript segment count mismatch: {len(segment)} segments but {len(texts)} texts."
        )

    docs, buf, start, end = [], [], None, None

    def flush():
        if buf:
            docs.append(Document(
                page_content=" ".join(buf),
                metadata={"start": float(start), "end": float(end)},
            ))

    for seg, text in zip(segment, texts):
        normalized = "" if text is None else str(text)
        if buf and sum(len(b) for b in buf) + len(normalized) > max_chars:
            flush()
            buf, start, end = [], None, None
        if start is None:
            start = seg.start
        buf.append(normalized)
        end = seg.end

    flush()
    return docs

def build_vector_store(docs: list[Document], embeddings) -> Chroma:
    """One fresh collection per meeting, so the chats never mix between meetings."""
    return Chroma.from_documents(
        docs,
        embeddings,
        collection_name=f"meeting_{uuid.uuid4().hex}",
    )

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite follow-up questions so they make sense on their own. "
     "Using the chat history, replace every pronoun or vague reference "
     "('that', 'it', 'this', 'he', 'she', 'they', 'the bug') with the specific thing "
     "it refers to. Only return the question unchanged if it has no such references. "
     "Output ONLY the rewritten question, nothing else.\n\n"
     "Example:\n"
     "History: user asked 'Who is fixing the login bug?'\n"
     "Latest question: 'When is that due?'\n"
     "Output: When is the login bug fix due?"),
    MessagesPlaceholder("history"),
    ("human", "Latest question: {question}\nOutput:"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a meeting using ONLY the transcript excerpts below. "
     "Each excerpt starts with its timestamp in [brackets]. "
     "Use the chat history to understand what words like 'that' or 'it' refer to. "
     "Answer only what was asked, concisely, and cite timestamps as [mm:ss]. "
     "If the answer is not in the excerpts, say you couldn't find it in the meeting.\n\n"
     "Excerpts:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

def to_messages(history: list[dict]):
    msgs = []
    for m in history[-6:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    return msgs

def format_docs(docs: list[Document]) -> str:
    docs = sorted(docs, key=lambda d: d.metadata["start"])
    return "\n\n".join(
        f"[{format_timestamp(d.metadata['start'])}] {d.page_content}" for d in docs
    )

def ask(vectorstore, question, history, k=4):
    rewrite_llm = get_llm(temperature=0)
    llm = get_llm(temperature=0.2)

    if history:
        rewrite = REWRITE_PROMPT | rewrite_llm | StrOutputParser()
        standalone = rewrite.invoke(
            {"history": to_messages(history), "question": question}
        ).strip()
    else:
        standalone = question

    docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(standalone)

    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = answer_chain.invoke({
        "context": format_docs(docs),
        "history": to_messages(history),
        "question": standalone,
    })
    return answer, docs