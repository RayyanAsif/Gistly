from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

def get_llm(temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("MODEL"),
        temperature=temperature,
        api_key=os.getenv("API_KEY"),
    )

class ActionItem(BaseModel):
    task: str = Field(description="What needs to be done.")
    owner: str = Field(default="Unassigned", description="Person Responsible.")
    deadline: str = Field(default="Unassigned", description="Due data if mentioned.")

class ActionItems(BaseModel):
    items: list[ActionItem]

class Decisions(BaseModel):
    decisions: list[str]

class Translations(BaseModel):
    translations: list[str]

def translate_to_english(texts: list[str], batch_size: int = 10) -> list[str]:
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You translate transcripts (Urdu, Hindi, Roman Urdu or Hinglish) into natural English. "
        "The transcript comes from speech recognition and contains errors. English loanwords "
        "(like 'birthday', 'surprise', 'meeting', 'sorry') are often misheard and written as "
        "similar-sounding Urdu words. Use the surrounding lines to infer what was actually "
        "said, and translate that. Keep character and person names as they are, do not "
        "translate them into common words. Translate the provided transcript segments; do "
        "not ask for more text. Return exactly one translation per segment, in the same order. "
        "Return the translations as valid JSON using the translations array."),
        ("human", "Translate these transcript segments:\n{lines}"),
    ])
    chain = prompt | llm.with_structured_output(Translations, method="json_mode")

    out = [""] * len(texts)
    nonempty = [(i, text) for i, text in enumerate(texts) if text.strip()]
    for start in range(0, len(nonempty), batch_size):
        batch = nonempty[start: start + batch_size]
        result = chain.invoke({"lines": "\n".join(text for _, text in batch)})
        if len(result.translations) != len(batch):
            raise ValueError(
                "Translation response count did not match the transcript segment count."
            )
        for (index, _), translation in zip(batch, result.translations):
            if not translation.strip():
                raise ValueError("Translation response contained an empty segment.")
            out[index] = translation.strip()
    return out

def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=500)

def summarize(transcript: str) -> str:
    llm = get_llm()
    chunks = _splitter().split_text(transcript)

    map_prompt = ChatPromptTemplate.from_template(
        "Summarize this part of a meeting transcript. Start with a short overview "
        "paragraph, followed by concise key bullet points:\n\n{text}"
    )

    reduce_prompt = ChatPromptTemplate.from_template(
        "Below are partial summaries of one meeting. Combine them into a single clear "
        "summary with a short overview paragraph followed by key bullet points.\n\n{text}"
    )

    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    if len(chunks) == 1:
        return map_chain.invoke({"text": chunks[0]})

    partials = map_chain.batch(
        [{"text": c} for c in chunks], config={"max_concurrency": 2}
    )

    return reduce_chain.invoke({"text": "\n\n".join(partials)})

def extract_action_items(transcript: str) -> list[ActionItem]:
    llm = get_llm(temperature=0).with_structured_output(
        ActionItems,
        method="json_mode",
    )
    prompt = ChatPromptTemplate.from_template(
        "Extract action items actually assigned or agreed upon in this "
        "meeting transcript. If none exist, return an empty items list.\n"
        "Return only valid JSON matching this schema:\n"
        '{{"items": [{{"task": "string", "owner": "string", '
        '"deadline": "string"}}]}}\n\n'
        "Transcript:\n{text}"
    )
    chain = prompt | llm 
    chunks = _splitter().split_text(transcript)

    results = chain.batch([{"text": c} for c in chunks], config={"max_concurrency": 2})

    seen, merged = set(), []

    for r in results:
        for item in r.items:
            key = item.task.lower().strip()
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return merged

def extract_decisions(transcript: str) -> list[str]:
    llm = get_llm(temperature=0).with_structured_output(
        Decisions,
        method="json_mode",
    )
    prompt = ChatPromptTemplate.from_template(
        "Extract only decisions explicitly finalized in this meeting "
        "transcript. Do not include suggestions or unresolved proposals. "
        "If none were finalized, return an empty decisions list.\n"
        "Return only valid JSON matching this schema:\n"
        '{{"decisions": ["string"]}}\n\n'
        "Transcript:\n{text}"
    )
    chain = (prompt | llm)
    chunks = _splitter().split_text(transcript)
    results = chain.batch([{"text": c} for c in chunks], config={"max_concurrency": 2})

    seen, merged = set(), []
    for r in results:
        for d in r.decisions:
            if d.lower().strip() not in seen:
                seen.add(d.lower().strip())
                merged.append(d)
    return merged

def analyze(transcript: str) -> dict:
    return {
        "summary": summarize(transcript),
        "action_items": extract_action_items(transcript),
        "decisions": extract_decisions(transcript),
    }