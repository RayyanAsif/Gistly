import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from utils.transcribe import format_timestamp

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/Nirmala.ttf"),
    FONT_DIR / "NotoSans-Regular.ttf",
    Path("C:/Windows/Fonts/arial.ttf"),
]
DEVANAGARI_FONT = FONT_DIR / "NotoSansDevanagari-Regular.ttf"


def clean_markdown(text: str) -> str:
    """Strip markdown and LaTeX markers so they don't show up as raw symbols."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)          
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)       
    text = re.sub(r"^\s*[\*\-]\s+", "• ", text, flags=re.M) 
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.M)       
    text = text.replace("\\(", "").replace("\\)", "")        
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _transcript_lines(segments, texts) -> list[str]:
    return [f"[{format_timestamp(s.start)}] {t}" for s, t in zip(segments, texts)]


def build_txt(title, result, segments, texts) -> str:
    out = [
        title,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 60,
        clean_markdown(result["summary"]),
        "",
        "ACTION ITEMS",
        "-" * 60,
    ]
    if result["action_items"]:
        for a in result["action_items"]:
            out.append(f"• {a.task}  (Owner: {a.owner}, Deadline: {a.deadline})")
    else:
        out.append("None found.")

    out += ["", "KEY DECISIONS", "-" * 60]
    if result["decisions"]:
        out += [f"• {d}" for d in result["decisions"]]
    else:
        out.append("None found.")

    out += ["", "TRANSCRIPT", "-" * 60, *_transcript_lines(segments, texts)]
    return "\n".join(out)

def _new_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    main = next((p for p in FONT_CANDIDATES if p.exists()), None)
    if main is None:
        raise FileNotFoundError(
            "No Unicode font found. Put NotoSans-Regular.ttf in the fonts/ folder."
        )
    pdf.add_font("Main", "", str(main))
    if DEVANAGARI_FONT.exists():
        pdf.add_font("Deva", "", str(DEVANAGARI_FONT))
        pdf.set_fallback_fonts(["Deva"])

    try:
        pdf.set_text_shaping(True)
    except Exception:
        pass
    return pdf


def _heading(pdf: FPDF, text: str):
    pdf.ln(3)
    pdf.set_font("Main", size=14)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Main", size=11)


def _body(pdf: FPDF, text: str):
    pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")


def build_pdf(title, result, segments, texts) -> bytes:
    pdf = _new_pdf()

    pdf.set_font("Main", size=20)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Main", size=9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    _heading(pdf, "Summary")
    _body(pdf, clean_markdown(result["summary"]))

    _heading(pdf, "Action Items")
    if result["action_items"]:
        for a in result["action_items"]:
            _body(pdf, f"• {a.task}")
            pdf.set_text_color(110, 110, 110)
            pdf.set_font("Main", size=9)
            _body(pdf, f"    Owner: {a.owner}   |   Deadline: {a.deadline}")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Main", size=11)
            pdf.ln(1)
    else:
        _body(pdf, "None found.")

    _heading(pdf, "Key Decisions")
    if result["decisions"]:
        for d in result["decisions"]:
            _body(pdf, f"• {d}")
    else:
        _body(pdf, "None found.")

    pdf.add_page()
    _heading(pdf, "Transcript")
    pdf.set_font("Main", size=9)
    for line in _transcript_lines(segments, texts):
        _body(pdf, line)

    return bytes(pdf.output())