import logging
import os
from pathlib import Path
import streamlit as st
from utils.audio import convert_uploaded_file, download_youtube_audio
from utils.chains import analyze, translate_to_english
from utils.rag import ask, build_document, build_vector_store, get_embeddings
from utils.transcribe import format_timestamp, load_model, transcribe
from utils.export import build_pdf, build_txt

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️", layout="wide")

LANGUAGES = {"Auto-detect": None, "English":"en", "Hindi": "hi", "Urdu":"ur"}

@st.cache_resource(show_spinner="Loading Whisper model...")
def cached_whisper(size: str):
    return load_model(size)

@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embeddings():
    return get_embeddings()

for key, default in {"data": None, "chat":[]}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def run_pipeline(source, model_size, language, terms, title):
    audio_path = None
    try: 
        with st.status("Processing meeting...", expanded=True) as status:
            st.write("1/5 Getting audio...")
            if isinstance(source, str):
                audio_path = download_youtube_audio(source)
            else:
                audio_path = convert_uploaded_file(source.getvalue(), source.name)

            st.write(f"2/5 Transcribing with Whisper ({model_size})... this will take some time")

            model = cached_whisper(model_size)
            segments, detected = transcribe(
                model, audio_path, language=language, prompt=terms or None
            )   
            if not segments:
                status.update(label="No speech found in this audio.", state="error")
                return 

            original = [s.text for s in segments]

            if detected != "en":
                st.write(f"3/5 translating from '{detected}' to English...")
                texts = translate_to_english(original)
            else:
                st.write("3/5 Already English, skipping translation")
                texts = original

            if len(segments) != len(texts):
                raise ValueError(
                    f"Transcript processing mismatch: {len(segments)} segments but {len(texts)} texts."
                )

            st.write("4/5 Summarizing, extracting action items and decisions...")

            result = analyze("\n".join(texts))

            st.write("5/5 Building search index and export files...")
            docs = build_document(segments, texts)
            vectorstore = build_vector_store(docs, cached_embeddings())

            txt = build_txt(title, result, segments, texts)
            try:
                pdf = build_pdf(title, result, segments, texts)
            except Exception:
                logger.exception("PDF export failed")
                pdf = None
                st.warning(
                    "PDF export is unavailable for this meeting. "
                    "You can still download the TXT export."
                )

            st.session_state.data = {
                "title": title,
                "language": detected,
                "segments": segments,
                "original": original,
                "texts": texts,
                "result": result,
                "vectorstore": vectorstore,
                "txt": txt,
                "pdf": pdf,
            } 
            st.session_state.chat = []
            status.update(label="Done!", state="complete", expanded=False)
    except Exception:
        logger.exception("Meeting processing failed")
        st.error("We couldn't process the meeting. Please check the input and try again.")
    finally:
        if audio_path and Path(audio_path).exists():
            try:
                os.remove(audio_path)
            except OSError:
                logger.exception("Failed to remove temporary audio file")


with st.sidebar:
    st.title("🎙️ Meeting Assistant")
    mode = st.radio("Input", ["Youtube URL", "Upload file"], horizontal=True)

    source = None
    if mode == "Youtube URL":
        url = st.text_input("Youtube URL")
        source = url.strip() or None
    else:
        source = st.file_uploader(
            "Audio or video file",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg", "flac"],
        )

    title = st.text_input("Meeting title", value="Meeting Notes")
    language_label = st.selectbox("Spoken language", list(LANGUAGES))
    model_size = st.selectbox(
        "Whisper model", ["tiny", "base", "small", "medium"], index = 2,
        help="Bigger = more accurate but slower in CPU. Use 'small' or higher for Urdu."
    )
    terms = st.text_input(
        "Names and terms (optional)",
        placeholder="e.g Rayyan, Asif, Q1 roadmap",
        help="Help whisper spell names and jargon correctly.",
    )
    if st.button(
        "Process meeting", type="primary", disabled=source is None,
        use_container_width=True,
    ):
        run_pipeline(source, model_size, LANGUAGES[language_label], terms, title)

data = st.session_state.data

if data is None:
    st.header("AI Meeting Assistant")
    st.write(
        "Add a YouTube link or upload a recording in the sidebar, then press "
        "**Process meeting**. You'll get a summary, action items, key decisions, "
        "a searchable chat, and PDF/TXT export."
    )
    st.stop()

st.header(data["title"])
st.caption(f"Detected Language: {data['language']} -- {len(data['segments'])} segments")

tab_summary, tab_actions, tab_chat, tab_transcript, tab_export = st.tabs(
    ["Summary", "Action items and decision", "Chat", "Transcript", "Export"]
)

with tab_summary:
    st.markdown(data["result"]["summary"])

with tab_actions:
    st.subheader("Action items")
    items = data["result"]["action_items"]

    if items:
        st.table([
            {"Task": a.task, "Owner": a.owner, "Deadline": a.deadline} for a in items])

    else:
        st.info("No action items found.")

    st.subheader("Key decisions")
    if data["result"]["decisions"]:
        for d in data["result"]["decisions"]:
            st.markdown(f"- {d}")
    else:
        st.info("No decision found.")

with tab_chat:
    st.caption("Ask questions about this meeting. Answer came only from the transcript.")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    for ts, text in m["sources"]:
                        st.markdown(f"**[{ts}]** {text}")

    question = st.chat_input("Ask about the meeting...")
    if question:
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.chat]

        st.session_state.chat.append({"role": "user","content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, docs = ask(data["vectorstore"], question, history)
                except Exception:
                    logger.exception("Meeting question-answering failed")
                    answer, docs = (
                        "Sorry, I couldn't answer that right now. Please try again.",
                        [],
                    )
            st.markdown(answer)

            sources = [
                (format_timestamp(d.metadata["start"]), d.page_content)
                for d in sorted(docs, key=lambda d: d.metadata["start"])
            ]
            if sources:
                with st.expander("Sources"):
                    for ts, text in sources:
                        st.markdown(f"**[{ts}]** {text}")

        st.session_state.chat.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

with tab_transcript:
    show_original = False
    if data["language"] != "en":
        show_original = st.toggle("Show original language")
    lines = data["original"] if show_original else data["texts"]
    for seg, line in zip(data["segments"], lines):
        st.markdown(f"**[{format_timestamp(seg.start)}]** {line}")

with tab_export:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in data["title"])
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download TXT", data["txt"], file_name=f"{safe_name}.txt",
        mime="text/plain", use_container_width=True,
    )
    if data["pdf"]:
        c2.download_button(
            "Download PDF", data["pdf"], file_name=f"{safe_name}.pdf",
            mime="application/pdf", use_container_width=True,
        )
    else:
        c2.button("PDF unavailable", disabled=True, use_container_width=True)
    if data["language"] != "en":
        original_txt = "\n".join(
            f"[{format_timestamp(s.start)}] {t}"
            for s, t in zip(data["segments"], data["original"])
        )
        c3.download_button(
            "Original transcript (TXT)", original_txt,
            file_name=f"{safe_name}_original.txt", mime="text/plain",
            use_container_width=True,
        )
 