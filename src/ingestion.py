import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from .models import Segment

# Patterns that identify interviewer vs participant speaker labels
_INTERVIEWER_RE = re.compile(
    r'^(INTERVIEWER|FACILITATOR|MODERATOR|I|Q)\s*[:]\s*', re.IGNORECASE
)
_SPEAKER_RE = re.compile(
    r'^([A-Z][A-Z0-9_]{0,10})\s*[:]\s*', re.IGNORECASE
)


def _load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".pdf":
        import PyPDF2
        pages = []
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)
    raise ValueError(f"Unsupported file type: {suffix}")


def _detect_speaker_turns(text: str) -> Optional[List[Tuple[str, str]]]:
    """Return list of (speaker, utterance) if text has turn-by-turn format."""
    lines = text.splitlines()
    turns: List[Tuple[str, str]] = []
    cur_speaker: Optional[str] = None
    cur_lines: List[str] = []

    for line in lines:
        m = _SPEAKER_RE.match(line.strip())
        if m:
            if cur_speaker and cur_lines:
                turns.append((cur_speaker, " ".join(cur_lines).strip()))
            cur_speaker = m.group(1).upper()
            cur_lines = [line[m.end():].strip()]
        elif cur_speaker is not None:
            cur_lines.append(line.strip())

    if cur_speaker and cur_lines:
        turns.append((cur_speaker, " ".join(cur_lines).strip()))

    return turns if len(turns) >= 4 else None


def _chunks(words: List[str], size: int) -> List[List[str]]:
    return [words[i : i + size] for i in range(0, len(words), size)]


def _segments_from_turns(
    turns: List[Tuple[str, str]], source: str, min_words: int, max_words: int
) -> List[Segment]:
    segs: List[Segment] = []
    pos = 0
    for speaker, text in turns:
        text = text.strip()
        if not text:
            continue
        is_interviewer = bool(_INTERVIEWER_RE.match(speaker + ":"))
        participant = None if is_interviewer else speaker
        words = text.split()
        if len(words) < min_words:
            continue
        for chunk in _chunks(words, max_words):
            if len(chunk) < min_words:
                continue
            segs.append(
                Segment(
                    id=str(uuid.uuid4())[:8],
                    source_file=source,
                    participant=participant,
                    text=" ".join(chunk),
                    position=pos,
                    word_count=len(chunk),
                )
            )
            pos += 1
    return segs


def _segments_from_paragraphs(
    text: str, source: str, min_words: int, max_words: int
) -> List[Segment]:
    segs: List[Segment] = []
    pos = 0
    paras = re.split(r"\n\s*\n", text)
    buf_words: List[str] = []

    for para in paras:
        para = para.strip()
        if not para:
            continue
        buf_words.extend(para.split())
        if len(buf_words) >= max_words:
            for chunk in _chunks(buf_words, max_words):
                if len(chunk) >= min_words:
                    segs.append(
                        Segment(
                            id=str(uuid.uuid4())[:8],
                            source_file=source,
                            participant=None,
                            text=" ".join(chunk),
                            position=pos,
                            word_count=len(chunk),
                        )
                    )
                    pos += 1
            buf_words = []

    if len(buf_words) >= min_words:
        segs.append(
            Segment(
                id=str(uuid.uuid4())[:8],
                source_file=source,
                participant=None,
                text=" ".join(buf_words),
                position=pos,
                word_count=len(buf_words),
            )
        )
    return segs


def load_transcript(
    path: Path, min_words: int = 30, max_words: int = 400
) -> List[Segment]:
    text = _load_text(path)
    source = path.name
    turns = _detect_speaker_turns(text)
    if turns:
        return _segments_from_turns(turns, source, min_words, max_words)
    return _segments_from_paragraphs(text, source, min_words, max_words)


def load_all_transcripts(
    directory: Path, min_words: int = 30, max_words: int = 400
) -> List[Segment]:
    supported = {".txt", ".docx", ".pdf"}
    all_segs: List[Segment] = []
    files = sorted(f for f in directory.iterdir() if f.suffix.lower() in supported)
    for fp in files:
        try:
            segs = load_transcript(fp, min_words, max_words)
            all_segs.extend(segs)
        except Exception as exc:
            print(f"Warning: skipping {fp.name}: {exc}")
    return all_segs
