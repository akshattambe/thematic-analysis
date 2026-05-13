"""
Theme synthesis using claude-opus-4-7 with adaptive thinking.
Two-stage: (1) cluster all codes → provisional themes, (2) synthesise each theme.
"""
import json
import re
from collections import Counter
from typing import Callable, Dict, List, Optional

import anthropic

from .models import CodedSegment, Quote, Theme

_CLUSTER_TMPL = """\
You are an expert qualitative researcher using Braun & Clarke (2006) thematic analysis.

You have coded {n_segs} interview segments from {n_files} transcript(s) and produced the \
following codes (format: "code — ×frequency"):

{codes_block}

Task: group these codes into {min_t}–{max_t} coherent, analytically distinct themes.

Rules:
- Each theme must have at least 3 codes
- Themes should be conceptually distinct (minimal overlap)
- Group by underlying meaning, not just surface wording
- Aim for themes rich enough to sustain a thesis discussion

Return ONLY valid JSON:
{{"themes": [{{"provisional_name": "...", "codes": ["code1", "code2", ...]}}]}}"""

_SYNTH_TMPL = """\
You are an expert qualitative researcher writing up a theme for an academic thesis.

Provisional theme name: {pname}

Codes in this theme:
{codes_block}

Most relevant data segments:
{segs_block}

Write a richly developed theme suitable for a thesis chapter. Return ONLY valid JSON:
{{
  "name": "Refined theme name (3–8 words, evocative and specific)",
  "description": "Two to three paragraphs: (1) what this theme captures and its boundaries, \
(2) analytical significance and how it relates to the research purpose, \
(3) any internal tensions or sub-patterns. Academic register, thesis-ready.",
  "representative_quotes": [
    {{"text": "exact quote from the data", "source_file": "filename", "participant": "label or null"}}
  ]
}}"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    for fence in ["```json", "```"]:
        if fence in raw:
            raw = raw.split(fence, 1)[1].split("```")[0]
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _top_segments(codes: List[str], all_cs: List[CodedSegment], n: int = 8) -> List[CodedSegment]:
    code_set = {c.lower().strip() for c in codes}
    scored = [
        (len({c.lower().strip() for c in cs.codes} & code_set), cs)
        for cs in all_cs
    ]
    scored = [(s, cs) for s, cs in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    return [cs for _, cs in scored[:n]]


def _cluster_codes(
    coded_segs: List[CodedSegment], min_t: int, max_t: int
) -> List[Dict]:
    counter: Counter = Counter()
    for cs in coded_segs:
        for c in cs.codes:
            counter[c.lower().strip()] += 1

    min_freq = 2 if len(counter) > 30 else 1
    sig = {c: f for c, f in counter.items() if f >= min_freq}

    codes_block = "\n".join(
        f"  {c} — ×{f}"
        for c, f in sorted(sig.items(), key=lambda x: -x[1])
    )

    n_files = len({cs.segment.source_file for cs in coded_segs})
    prompt = _CLUSTER_TMPL.format(
        n_segs=len(coded_segs),
        n_files=n_files,
        codes_block=codes_block,
        min_t=min_t,
        max_t=max_t,
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    data = _parse_json(text)
    return data.get("themes", [])


def _synthesise_theme(
    provisional: Dict, all_cs: List[CodedSegment], theme_id: int
) -> Theme:
    codes = provisional.get("codes", [])
    pname = provisional.get("provisional_name", f"Theme {theme_id}")

    relevant = _top_segments(codes, all_cs, n=8)
    segs_block = ""
    for i, cs in enumerate(relevant[:5], 1):
        s = cs.segment
        speaker = f" ({s.participant})" if s.participant else ""
        snippet = s.text[:350] + ("…" if len(s.text) > 350 else "")
        segs_block += (
            f"\n[{i}] {s.source_file}{speaker}\n"
            f'"{snippet}"\n'
            f"Codes: {', '.join(cs.codes)}\n"
        )

    codes_block = "\n".join(f"  - {c}" for c in codes)
    prompt = _SYNTH_TMPL.format(
        pname=pname, codes_block=codes_block, segs_block=segs_block
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2500,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    data = _parse_json(text)

    quotes: List[Quote] = []
    for q in data.get("representative_quotes", [])[:5]:
        if isinstance(q, dict) and q.get("text"):
            quotes.append(
                Quote(
                    text=q["text"],
                    source_file=q.get("source_file", ""),
                    participant=q.get("participant"),
                )
            )

    # Fall back to actual segment excerpts if LLM gave none
    if not quotes:
        for cs in relevant[:3]:
            quotes.append(
                Quote(
                    text=cs.segment.text[:300] + ("…" if len(cs.segment.text) > 300 else ""),
                    source_file=cs.segment.source_file,
                    participant=cs.segment.participant,
                )
            )

    theme_segs = _top_segments(codes, all_cs, n=200)
    return Theme(
        id=theme_id,
        name=data.get("name", pname),
        description=data.get("description", ""),
        codes=codes,
        representative_quotes=quotes,
        coded_segments=theme_segs,
    )


def build_themes(
    coded_segs: List[CodedSegment],
    min_themes: int = 4,
    max_themes: int = 10,
    on_progress: Optional[Callable[[str], None]] = None,
    on_theme_ready: Optional[Callable] = None,
) -> List[Theme]:
    if on_progress:
        on_progress("Clustering codes into themes…")

    clusters = _cluster_codes(coded_segs, min_themes, max_themes)
    if not clusters:
        raise RuntimeError("Claude returned no theme clusters — check your API key and data.")

    themes: List[Theme] = []
    for i, cluster in enumerate(clusters, 1):
        pname = cluster.get("provisional_name", f"Theme {i}")
        if on_progress:
            on_progress(f"Synthesising theme {i}/{len(clusters)}: {pname}")
        theme = _synthesise_theme(cluster, coded_segs, i)
        themes.append(theme)
        if on_theme_ready:
            on_theme_ready(theme)

    return themes
