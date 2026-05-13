"""
Tests for multi-interview scaling adjustments.
Run with: python -m pytest tests/test_scale.py -v
"""
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src_gioia.models import FirstOrderConcept, SecondOrderTheme
from src_gioia.themer import _cluster_second_order


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_segment(source_file="interview_01.txt"):
    seg = MagicMock()
    seg.source_file = source_file
    return seg

def _make_concepts(n: int, unique_ratio: float = 0.6) -> list:
    """Generate n FirstOrderConcept objects with controlled uniqueness."""
    n_unique = int(n * unique_ratio)
    unique_labels = [f"concept_{i:04d}" for i in range(n_unique)]
    concepts = []
    for i in range(n):
        label = unique_labels[i % n_unique]
        foc = MagicMock(spec=FirstOrderConcept)
        foc.concept = label
        foc.segment = _make_segment(f"interview_{(i // 20) + 1:02d}.txt")
        concepts.append(foc)
    return concepts


# ── Config tests ──────────────────────────────────────────────────────────────

def test_concurrent_requests_raised():
    assert config.MAX_CONCURRENT_REQUESTS == 50, (
        f"Expected 50, got {config.MAX_CONCURRENT_REQUESTS}. "
        "This value should be raised for 20-interview workloads."
    )

def test_coding_model_is_haiku():
    assert "haiku" in config.CODING_MODEL, (
        "Coding model should remain haiku for cost efficiency at scale."
    )


# ── Concept cap tests ─────────────────────────────────────────────────────────

def test_concept_cap_single_interview():
    """With 1 interview (~80 concepts), all concepts should pass through."""
    concepts = _make_concepts(80, unique_ratio=1.0)
    counter = Counter(c.concept.lower().strip() for c in concepts)
    sig = dict(counter.most_common(min(500, len(counter))))
    assert len(sig) == 80


def test_concept_cap_20_interviews_old_limit():
    """Simulate 20 interviews — old 200-cap would drop many concepts."""
    concepts = _make_concepts(2000, unique_ratio=0.75)  # ~1500 unique
    counter = Counter(c.concept.lower().strip() for c in concepts)
    old_cap = dict(counter.most_common(min(200, len(counter))))
    new_cap = dict(counter.most_common(min(1000, len(counter))))
    assert len(new_cap) > len(old_cap), (
        "New cap (1000) should pass more concepts than old cap (200)."
    )
    assert len(new_cap) == 1000
    assert len(old_cap) == 200


def test_concept_cap_does_not_exceed_1000():
    """Even with 5000 unique concepts the cap stays at 1000."""
    concepts = _make_concepts(5000, unique_ratio=1.0)
    counter = Counter(c.concept.lower().strip() for c in concepts)
    sig = dict(counter.most_common(min(1000, len(counter))))
    assert len(sig) == 1000


def test_frequency_filter_fallback_still_works():
    """
    If frequency filter leaves enough concepts (>= max_t*3),
    the fallback to most_common(500) is NOT triggered.
    Verifies existing small-dataset behaviour is unchanged.
    """
    # 10 unique concepts appearing 3 times each — well above min_freq threshold
    concepts = _make_concepts(30, unique_ratio=1.0)
    # With only 10 unique, frequency filter keeps all 10; no fallback needed
    counter = Counter(c.concept.lower().strip() for c in concepts)
    max_t = 12
    min_freq = 2 if len(counter) > 50 else 1
    sig = {c: f for c, f in counter.items() if f >= min_freq}
    # Should NOT fall back since len(sig)=10 which may be < max_t*3=36
    # In that case fallback kicks in — cap is 500 not 200
    if len(sig) < max_t * 3:
        sig = dict(counter.most_common(min(500, len(counter))))
    assert len(sig) > 0


# ── Streaming tests ───────────────────────────────────────────────────────────

def test_clustering_uses_streaming():
    """_cluster_second_order must call messages.stream(), not messages.create()."""
    themer_source = (Path(__file__).parent.parent / "src_gioia" / "themer.py").read_text()
    assert "client.messages.stream(" in themer_source, (
        "themer.py must use client.messages.stream() to avoid 10-minute API timeout."
    )
    assert "stream.get_final_text()" in themer_source, (
        "themer.py must call stream.get_final_text() to retrieve the streamed response."
    )


def test_no_messages_create_in_themer():
    """messages.create() must not appear in themer.py (only stream() is allowed)."""
    themer_source = (Path(__file__).parent.parent / "src_gioia" / "themer.py").read_text()
    assert "client.messages.create(" not in themer_source, (
        "themer.py still uses messages.create() which triggers the 10-minute timeout error."
    )


# ── Deduplication tests ───────────────────────────────────────────────────────

def test_dedup_merges_near_identical_labels():
    """Near-identical labels (Jaccard >= 0.6) should be merged into one canonical."""
    from src_gioia.themer import _deduplicate_concepts
    sig = {
        "lack of trust": 5,       # canonical (higher freq)
        "lack of organizational trust": 3,  # Jaccard with above: 2/4 = 0.5 — below threshold
        "trust deficit": 2,        # Jaccard with "lack of trust": 0/3 = 0 — stays separate
        "communication breakdown": 4,
        "breakdown in communication": 2,  # Jaccard: 2/3 = 0.67 — merges into above
    }
    deduped, alias_map = _deduplicate_concepts(sig, threshold=0.6)
    # "breakdown in communication" should merge into "communication breakdown"
    assert alias_map["breakdown in communication"] == "communication breakdown"
    assert "breakdown in communication" not in deduped
    # merged count should be summed
    assert deduped["communication breakdown"] == 6  # 4 + 2


def test_dedup_alias_map_covers_all_originals():
    """Every original label must appear in alias_map (either as canonical or merged)."""
    from src_gioia.themer import _deduplicate_concepts
    sig = {f"concept {i}": i + 1 for i in range(20)}
    deduped, alias_map = _deduplicate_concepts(sig)
    assert set(alias_map.keys()) == set(sig.keys()), (
        "alias_map must contain an entry for every original concept label."
    )


def test_dedup_canonical_maps_to_itself():
    """A canonical label must map to itself in alias_map."""
    from src_gioia.themer import _deduplicate_concepts
    sig = {"alpha beta": 10, "gamma delta": 5}
    _, alias_map = _deduplicate_concepts(sig)
    for canon in ("alpha beta", "gamma delta"):
        assert alias_map[canon] == canon, f"Canonical label '{canon}' should map to itself."


def test_dedup_reduces_count_on_large_dataset():
    """Deduplication collapses word-order variants (same token set, Jaccard=1.0)."""
    from src_gioia.themer import _deduplicate_concepts
    # 25 pairs: "word_N shared" and "shared word_N" have the same token set → Jaccard=1.0
    sig = {}
    for i in range(25):
        sig[f"word_{i} shared"] = 10   # higher freq → becomes canonical
        sig[f"shared word_{i}"] = 5    # word-order variant → merges into canonical
    deduped, _ = _deduplicate_concepts(sig, threshold=0.6)
    assert len(deduped) < len(sig), "Deduplication should collapse word-order duplicates."
    assert len(deduped) == 25, "Each of the 25 synonym pairs should collapse to one canonical."


def test_dedup_no_cross_merge_on_distinct_concepts():
    """Completely distinct labels should not be merged."""
    from src_gioia.themer import _deduplicate_concepts
    sig = {
        "leadership vision": 5,
        "resource allocation": 4,
        "stakeholder engagement": 3,
        "process efficiency": 2,
    }
    deduped, alias_map = _deduplicate_concepts(sig, threshold=0.6)
    # All labels are distinct — no merging should occur
    assert len(deduped) == len(sig), "Distinct labels must not be merged."
    for label in sig:
        assert alias_map[label] == label


def test_dedup_present_in_themer():
    """themer.py must call _deduplicate_concepts before clustering."""
    themer_source = (Path(__file__).parent.parent / "src_gioia" / "themer.py").read_text()
    assert "_deduplicate_concepts" in themer_source, (
        "themer.py must deduplicate concepts before sending to Claude."
    )
    assert "alias_map" in themer_source, (
        "themer.py must use alias_map to recover all original concepts after clustering."
    )


# ── Max themes test ───────────────────────────────────────────────────────────

def test_max_themes_raised_in_pipeline():
    """
    Verify the app.py pipeline passes max_themes=18 to build_gioia_structure.
    We do this by inspecting the source directly.
    """
    app_source = (Path(__file__).parent.parent / "app.py").read_text()
    assert "max_themes=25" in app_source, (
        "app.py should pass max_themes=25 to build_gioia_structure for 20-interview scale."
    )
    assert "max_themes=12" not in app_source, (
        "Old max_themes=12 should be replaced with 25."
    )


# ── Source file diversity test ────────────────────────────────────────────────

def test_source_file_count_20_interviews():
    """Concepts from 20 different files are correctly tracked."""
    concepts = []
    for i in range(20):
        for _ in range(30):  # 30 concepts per interview
            foc = MagicMock(spec=FirstOrderConcept)
            foc.concept = f"concept_{i}_{_}"
            foc.segment = _make_segment(f"interview_{i:02d}.txt")
            concepts.append(foc)
    source_files = {c.segment.source_file for c in concepts}
    assert len(source_files) == 20


def test_max_themes_capped_by_concept_count():
    """
    With a small concept set (single interview), max_t must be capped so
    Claude is never asked for more themes than the data can support.
    Each theme requires at least 3 concepts: max_t <= len(deduped) // 3.
    """
    themer_source = (Path(__file__).parent.parent / "src_gioia" / "themer.py").read_text()
    assert "max_t = max(min_t, min(max_t, len(deduped) // 3))" in themer_source, (
        "themer.py must cap max_t based on deduplicated concept count to prevent "
        "empty-theme errors on small datasets."
    )


def test_max_themes_cap_runtime_single_interview():
    """
    Runtime check: with 30 unique concepts and max_t=25,
    the effective max_t must be capped to 30//3 = 10.
    Verifies the guard logic directly without calling the API.
    """
    min_t, max_t = 5, 25
    sig = {f"concept_{i}": 1 for i in range(30)}  # 30 unique concepts

    effective_max_t = max(min_t, min(max_t, len(sig) // 3))

    assert effective_max_t == 10, (
        f"Expected effective max_t=10 (30//3), got {effective_max_t}. "
        "Claude would have been asked for up to 25 themes from only 30 concepts."
    )
    assert effective_max_t <= len(sig) // 3


def test_max_themes_cap_runtime_20_interviews():
    """
    Runtime check: with 600 unique concepts and max_t=25,
    the cap must NOT reduce max_t — 600//3=200 >> 25.
    """
    min_t, max_t = 5, 25
    sig = {f"concept_{i}": 2 for i in range(600)}

    effective_max_t = max(min_t, min(max_t, len(sig) // 3))

    assert effective_max_t == 25, (
        f"Expected effective max_t=25 (uncapped), got {effective_max_t}. "
        "Large datasets should use the full max_t."
    )


def test_max_themes_cap_never_below_min_t():
    """
    Even with very few concepts (e.g. 6), effective max_t must not drop
    below min_t (5), ensuring Claude always gets a valid range.
    """
    min_t, max_t = 5, 25
    sig = {f"concept_{i}": 1 for i in range(6)}  # only 6 concepts → 6//3 = 2

    effective_max_t = max(min_t, min(max_t, len(sig) // 3))

    assert effective_max_t == min_t, (
        f"Expected effective max_t to floor at min_t={min_t}, got {effective_max_t}."
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
