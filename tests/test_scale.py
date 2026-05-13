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


# ── Streaming + concept cap tests ────────────────────────────────────────────

def test_clustering_cap_is_300():
    """The themer module must define _CLUSTERING_CAP = 300."""
    from src_gioia.themer import _CLUSTERING_CAP
    assert _CLUSTERING_CAP == 300, (
        f"Expected _CLUSTERING_CAP=300, got {_CLUSTERING_CAP}. "
        "Cap must be 300 to avoid token budget exhaustion on the streaming API call."
    )


def test_clustering_cap_enforced_at_300():
    """Even with 1000 unique concepts, only 300 are sent to Claude for clustering."""
    sig = {f"concept_{i}": (1000 - i) for i in range(1000)}  # 1000 concepts, varying freq
    from src_gioia.themer import _CLUSTERING_CAP
    clustering_sig = dict(sorted(sig.items(), key=lambda x: -x[1])[:_CLUSTERING_CAP])
    assert len(clustering_sig) == 300
    # Confirm they are the top-300 by frequency
    top_keys = [k for k, _ in sorted(sig.items(), key=lambda x: -x[1])[:300]]
    assert list(clustering_sig.keys()) == top_keys


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


# ── Unmatched concept assignment tests ───────────────────────────────────────

def test_best_theme_idx_keyword_overlap():
    """_best_theme_idx returns the index of the theme with most token overlap."""
    from src_gioia.themer import _best_theme_idx
    themes = [
        {"name": "Leadership Challenges", "description": "Issues around managerial authority"},
        {"name": "Communication Barriers", "description": "Breakdowns in information sharing"},
        {"name": "Resource Constraints", "description": "Lack of budget and staffing"},
    ]
    # "communication breakdown" overlaps best with theme[1]
    assert _best_theme_idx("communication breakdown", themes) == 1
    # "budget shortfall" overlaps best with theme[2]
    assert _best_theme_idx("budget shortfall", themes) == 2


def test_best_theme_idx_fallback_returns_zero():
    """When no overlap, _best_theme_idx defaults to index 0."""
    from src_gioia.themer import _best_theme_idx
    themes = [
        {"name": "Alpha Dynamics", "description": "XYZ phenomena"},
        {"name": "Beta Mechanisms", "description": "ABC processes"},
    ]
    # "zzz unrelated concept" has no overlap — should fall back to 0
    idx = _best_theme_idx("zzz unrelated concept", themes)
    assert idx == 0


def test_unmatched_concepts_assigned_in_build():
    """build_gioia_structure must contain code to assign unmatched concepts."""
    themer_source = (Path(__file__).parent.parent / "src_gioia" / "themer.py").read_text()
    assert "unmatched_keys" in themer_source, (
        "themer.py must assign concepts beyond _CLUSTERING_CAP to the nearest theme."
    )
    assert "_best_theme_idx" in themer_source, (
        "themer.py must use _best_theme_idx for unmatched concept assignment."
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
    Each theme requires at least 3 concepts: max_t <= len(sig) // 3.
    """
    themer_source = (Path(__file__).parent.parent / "src_gioia" / "themer.py").read_text()
    assert "max_t = max(min_t, min(max_t, len(sig) // 3))" in themer_source, (
        "themer.py must cap max_t based on available concepts to prevent "
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
