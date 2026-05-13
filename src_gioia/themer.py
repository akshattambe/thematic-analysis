"""
Two-stage Gioia theme building using claude-opus-4-7 with adaptive thinking.

Stage 1: 1st-order concept labels → 2nd-order theoretical themes
Stage 2: 2nd-order themes → aggregate dimensions
Stage 3: Synthesise rich descriptions for each aggregate dimension
"""
import json
import logging
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
    for fence in ["```json", "```"]:
        if fence in raw:
            raw = raw.split(fence, 1)[1].split("```")[0].strip()
            break
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
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


def _deduplicate_concepts(
    sig: Dict[str, int], threshold: float = 0.6
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Merge near-identical concept labels using token-set Jaccard similarity.

    Two labels merge when their token-set Jaccard score >= threshold.
    The higher-frequency label becomes canonical; counts are summed.

    Returns:
        deduped:   {canonical_label: summed_count}
        alias_map: {every_original_label: its_canonical_label}
    """
    labels = sorted(sig.keys(), key=lambda k: -sig[k])  # highest-freq first → canonical
    deduped: Dict[str, int] = {}
    alias_map: Dict[str, str] = {}

    for label in labels:
        if label in alias_map:
            continue
        tokens_a = set(label.split())
        deduped[label] = sig[label]
        alias_map[label] = label
        for other in labels:
            if other in alias_map:
                continue
            tokens_b = set(other.split())
            union = tokens_a | tokens_b
            if not union:
                continue
            jaccard = len(tokens_a & tokens_b) / len(union)
            if jaccard >= threshold:
                deduped[label] += sig[other]
                alias_map[other] = label

    return deduped, alias_map


def _cluster_second_order(
    concepts: List[FirstOrderConcept], min_t: int, max_t: int
) -> Tuple[List[dict], Dict[str, str]]:
    """
    Cluster first-order concepts into second-order themes.

    Returns (raw_themes, alias_map) where alias_map maps every original
    concept label to its canonical (deduplicated) label.
    """
    counter: Counter = Counter(c.concept.lower().strip() for c in concepts)
    min_freq = 2 if len(counter) > 50 else 1
    sig = {c: f for c, f in counter.items() if f >= min_freq}
    if len(sig) < max_t * 3:
        sig = dict(counter.most_common(min(1000, len(counter))))

    # Deduplicate synonymous labels so Claude sees each semantic concept once.
    # Every original label maps back to a canonical via alias_map.
    deduped, alias_map = _deduplicate_concepts(sig)
    logger.info(
        "Deduplicated %d concepts → %d canonical labels (%.0f%% reduction)",
        len(sig), len(deduped), 100 * (1 - len(deduped) / max(len(sig), 1)),
    )

    # Cap max_t based on the deduplicated count (each theme needs ≥3 concepts)
    max_t = max(min_t, min(max_t, len(deduped) // 3))

    concepts_block = "\n".join(
        f"  {c} — \u00d7{f}"
        for c, f in sorted(deduped.items(), key=lambda x: -x[1])
    )
    n_files = len({c.segment.source_file for c in concepts})
    prompt = (
        _SECOND_ORDER_TMPL
        .replace("{n_concepts}", str(len(deduped)))
        .replace("{n_files}", str(n_files))
        .replace("{concepts_block}", concepts_block)
        .replace("{min_t}", str(min_t))
        .replace("{max_t}", str(max_t))
    )
    logger.info("Clustering %d canonical concepts into %d–%d themes", len(deduped), min_t, max_t)
    client = anthropic.Anthropic()
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=12000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        text = stream.get_final_text()

    parsed = _parse_json(text)
    themes = parsed.get("themes", [])
    if not themes:
        logger.error("No themes parsed. Raw response text: %s", text[:500])
    return themes, alias_map


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
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        text = stream.get_final_text()

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
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        text = stream.get_final_text()

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
    if on_progress:
        on_progress("Clustering concepts into 2nd-order themes…")

    raw_themes, alias_map = _cluster_second_order(concepts, min_themes, max_themes)
    if not raw_themes:
        raise RuntimeError("Claude returned no 2nd-order themes — check API key and data.")

    # Build label_map: original label → List[FirstOrderConcept]
    label_map: Dict[str, list] = {}
    for c in concepts:
        key = c.concept.lower().strip()
        label_map.setdefault(key, []).append(c)

    # Invert alias_map: canonical → [all original labels that merged into it]
    canonical_to_originals: Dict[str, list] = {}
    for orig, canon in alias_map.items():
        canonical_to_originals.setdefault(canon, []).append(orig)

    second_order: List[SecondOrderTheme] = []
    for i, rt in enumerate(raw_themes, 1):
        matched: List[FirstOrderConcept] = []
        for label in rt.get("concepts", []):
            canon = label.lower().strip()
            # Include every original concept that was deduplicated into this canonical label
            for orig in canonical_to_originals.get(canon, [canon]):
                matched.extend(label_map.get(orig, []))
        theme = SecondOrderTheme(
            id=i,
            name=rt.get("name", f"Theme {i}"),
            description=rt.get("description", ""),
            first_order_concepts=matched,
        )
        second_order.append(theme)
        if on_theme_ready:
            on_theme_ready(theme)

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

    for i, dim in enumerate(aggregate, 1):
        if on_progress:
            on_progress(f"Synthesising dimension {i}/{len(aggregate)}: {dim.name}")
        dim.description = _synthesise_dimension(dim)
        if on_dimension_ready:
            on_dimension_ready(dim)

    return second_order, aggregate
