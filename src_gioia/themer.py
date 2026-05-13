"""
Two-stage Gioia theme building using claude-opus-4-7 with adaptive thinking.

Stage 1: 1st-order concept labels → 2nd-order theoretical themes
Stage 2: 2nd-order themes → aggregate dimensions
Stage 3: Synthesise rich descriptions for each aggregate dimension
"""
import json
import re
from collections import Counter
from typing import Callable, List, Optional, Tuple

import anthropic

from .models import AggregateDimension, FirstOrderConcept, SecondOrderTheme

_SECOND_ORDER_TMPL = """\
You are conducting Gioia Methodology research (Gioia, Corley & Hamilton, 2013).

{n_concepts} first-order concepts have been generated from {n_files} transcript(s). \
Each concept is written in the INFORMANT'S OWN LANGUAGE. \
Concept frequencies (format: "concept — ×count"):

{concepts_block}

Task: Cluster these into {min_t}–{max_t} SECOND-ORDER THEORETICAL THEMES using RESEARCHER language.

Second-order themes must:
- Be written in ABSTRACT THEORETICAL language (NOT informant language)
- Capture a theoretically meaningful pattern across multiple concepts
- Each theme must include at least 3 concept labels
- Be conceptually distinct from one another
- Use the full concept labels exactly as they appear above

Return ONLY valid JSON:
{{"themes": [{{"name": "Theoretical Theme Name", \
"description": "One to two sentences: what theoretical pattern this theme captures.", \
"concepts": ["exact concept label 1", "exact concept label 2"]}}]}}"""

_AGGREGATE_TMPL = """\
You are conducting Gioia Methodology research.

You have {n_themes} second-order theoretical themes:

{themes_block}

Task: Group these into 2–4 AGGREGATE DIMENSIONS — the highest level of theoretical abstraction.

Aggregate dimensions:
- Overarching theoretical constructs subsuming multiple 2nd-order themes
- Written at the highest level of abstraction
- Each must encompass at least 2 second-order themes
- Represent the core theoretical contribution of the analysis

Return ONLY valid JSON:
{{"dimensions": [{{"name": "Aggregate Dimension Name", \
"description": "One to two sentences on the overarching theoretical significance.", \
"theme_ids": [1, 2]}}]}}"""

_DIM_SYNTH_TMPL = """\
You are writing up an aggregate dimension for a Gioia methodology dissertation chapter.

Dimension: {name}

Second-order themes within this dimension:
{themes_block}

Write a rich academic description (2–3 paragraphs) suitable for a dissertation:
(1) What overarching phenomenon this dimension captures and its boundaries
(2) How the 2nd-order themes relate and build on each other within this dimension
(3) Theoretical significance and contribution to the literature

Return ONLY valid JSON:
{{"description": "two to three paragraphs..."}}"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    # Strip markdown fences
    for fence in ["```json", "```"]:
        if fence in raw:
            raw = raw.split(fence, 1)[1].split("```")[0].strip()
            break
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Walk backwards from the last } to find a valid JSON object
    end = raw.rfind("}")
    while end >= 0:
        start = raw.rfind("{", 0, end + 1)
        if start < 0:
            break
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            end = raw.rfind("}", 0, end)
    return {}


def _cluster_second_order(
    concepts: List[FirstOrderConcept], min_t: int, max_t: int
) -> List[dict]:
    counter: Counter = Counter(c.concept.lower().strip() for c in concepts)
    # Apply frequency filter only when it leaves enough concepts to form themes.
    # Guard: always keep at least max_t*3 concepts so Claude has material to cluster.
    min_freq = 2 if len(counter) > 50 else 1
    sig = {c: f for c, f in counter.items() if f >= min_freq}
    if len(sig) < max_t * 3:
        # Fall back to top-N concepts by frequency (no filter)
        sig = dict(counter.most_common(min(200, len(counter))))

    concepts_block = "\n".join(
        f"  {c} — \u00d7{f}"
        for c, f in sorted(sig.items(), key=lambda x: -x[1])
    )
    n_files = len({c.segment.source_file for c in concepts})
    # Build prompt without .format() to avoid KeyError on user data containing { }
    prompt = (
        _SECOND_ORDER_TMPL
        .replace("{n_concepts}", str(len(sig)))
        .replace("{n_files}", str(n_files))
        .replace("{concepts_block}", concepts_block)
        .replace("{min_t}", str(min_t))
        .replace("{max_t}", str(max_t))
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return _parse_json(text).get("themes", [])


def _cluster_aggregate(themes: List[SecondOrderTheme]) -> List[dict]:
    themes_block = "\n".join(
        f"  [ID {t.id}] {t.name}: {t.description[:150]}"
        for t in themes
    )
    prompt = (
        _AGGREGATE_TMPL
        .replace("{n_themes}", str(len(themes)))
        .replace("{themes_block}", themes_block)
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=3000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return _parse_json(text).get("dimensions", [])


def _synthesise_dimension(dim: AggregateDimension) -> str:
    themes_block = "\n".join(
        f"  - {t.name}: {t.description[:200]}" for t in dim.second_order_themes
    )
    prompt = (
        _DIM_SYNTH_TMPL
        .replace("{name}", dim.name)
        .replace("{themes_block}", themes_block)
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return _parse_json(text).get("description", dim.description)


def build_gioia_structure(
    concepts: List[FirstOrderConcept],
    min_themes: int = 5,
    max_themes: int = 12,
    on_progress: Optional[Callable[[str], None]] = None,
    on_theme_ready: Optional[Callable] = None,
    on_dimension_ready: Optional[Callable] = None,
) -> Tuple[List[SecondOrderTheme], List[AggregateDimension]]:
    """
    Returns (second_order_themes, aggregate_dimensions).
    Raises RuntimeError if Claude returns no clusters.
    """
    # Stage 1: cluster concept labels → 2nd-order themes
    if on_progress:
        on_progress("Clustering concepts into 2nd-order themes…")

    raw_themes = _cluster_second_order(concepts, min_themes, max_themes)
    if not raw_themes:
        raise RuntimeError("Claude returned no 2nd-order themes — check API key and data.")

    # Map concept labels back to FirstOrderConcept objects
    label_map: dict = {}
    for c in concepts:
        key = c.concept.lower().strip()
        label_map.setdefault(key, []).append(c)

    second_order: List[SecondOrderTheme] = []
    for i, rt in enumerate(raw_themes, 1):
        matched: List[FirstOrderConcept] = []
        for label in rt.get("concepts", []):
            matched.extend(label_map.get(label.lower().strip(), []))
        theme = SecondOrderTheme(
            id=i,
            name=rt.get("name", f"Theme {i}"),
            description=rt.get("description", ""),
            first_order_concepts=matched,
        )
        second_order.append(theme)
        if on_theme_ready:
            on_theme_ready(theme)

    # Stage 2: cluster 2nd-order themes → aggregate dimensions
    if on_progress:
        on_progress("Building aggregate dimensions…")

    raw_dims = _cluster_aggregate(second_order)
    if not raw_dims:
        raise RuntimeError("Claude returned no aggregate dimensions.")

    theme_map = {t.id: t for t in second_order}
    aggregate: List[AggregateDimension] = []
    for i, rd in enumerate(raw_dims, 1):
        matched_themes = [theme_map[tid] for tid in rd.get("theme_ids", []) if tid in theme_map]
        dim = AggregateDimension(
            id=i,
            name=rd.get("name", f"Dimension {i}"),
            description=rd.get("description", ""),
            second_order_themes=matched_themes,
        )
        aggregate.append(dim)

    # Stage 3: synthesise rich descriptions for each dimension
    for i, dim in enumerate(aggregate, 1):
        if on_progress:
            on_progress(f"Synthesising dimension {i}/{len(aggregate)}: {dim.name}")
        dim.description = _synthesise_dimension(dim)
        if on_dimension_ready:
            on_dimension_ready(dim)

    return second_order, aggregate
