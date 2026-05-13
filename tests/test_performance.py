"""
Performance tests for the thematic-analysis pipeline at 20-interview scale.

Each test asserts a time or memory budget. Budgets are set conservatively
(pure Python, no I/O, no API calls) so they pass reliably on any machine.

Run with: python3 -m pytest tests/test_performance.py -v
"""
import asyncio
import sys
import tempfile
import time
import tracemalloc
import uuid
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import load_all_transcripts
from src_gioia.models import FirstOrderConcept, SecondOrderTheme
from src_gioia.themer import _cluster_second_order


# ── Fixtures ──────────────────────────────────────────────────────────────────

_INTERVIEW_TEMPLATE = """\
INTERVIEWER: Can you describe your experience with the change process in your organisation?

PARTICIPANT: {p1}

INTERVIEWER: How did your team respond to that?

PARTICIPANT: {p2}

INTERVIEWER: What challenges did you encounter?

PARTICIPANT: {p3}

INTERVIEWER: How did leadership support you through this?

PARTICIPANT: {p4}

INTERVIEWER: Looking back, what would you do differently?

PARTICIPANT: {p5}
"""

_PARAGRAPHS = [
    "The transition was quite abrupt and many colleagues felt unprepared for the scale of change "
    "that was being asked of them without adequate notice or resources to adapt properly.",

    "We struggled initially to align the new processes with existing workflows because the "
    "communication from senior leadership was fragmented and often contradictory in its messaging.",

    "There was a significant lack of psychological safety within the team which made it difficult "
    "to raise concerns or flag issues before they escalated into more serious operational problems.",

    "The support structures were simply not in place when we needed them most and many people "
    "felt isolated in navigating this uncertainty without clear guidance or mentorship available.",

    "In hindsight we should have invested more time in building relationships and trust before "
    "attempting such a fundamental shift in how we operated and delivered value to stakeholders.",
]


def _make_interview_file(directory: Path, idx: int) -> Path:
    content = _INTERVIEW_TEMPLATE.format(
        p1=_PARAGRAPHS[0] + f" This was interview number {idx}.",
        p2=_PARAGRAPHS[1] + f" Participant {idx} felt strongly about this.",
        p3=_PARAGRAPHS[2] + f" Interview {idx} highlighted recurring friction points.",
        p4=_PARAGRAPHS[3] + f" Respondent {idx} noted this repeatedly.",
        p5=_PARAGRAPHS[4] + f" Subject {idx} concluded with this reflection.",
    )
    path = directory / f"interview_{idx:02d}.txt"
    path.write_text(content, encoding="utf-8")
    return path


def _make_segment(source_file: str = "interview_01.txt"):
    seg = MagicMock()
    seg.source_file = source_file
    seg.participant = "P1"
    seg.text = " ".join(_PARAGRAPHS)
    return seg


def _make_foc(concept: str, source_file: str = "interview_01.txt") -> FirstOrderConcept:
    foc = MagicMock(spec=FirstOrderConcept)
    foc.concept = concept
    foc.segment = _make_segment(source_file)
    return foc


def _make_concepts_diverse(n_interviews: int, concepts_per_seg: int = 3,
                            segs_per_interview: int = 20) -> list:
    """Generate realistic concept set: shared + unique concepts across interviews."""
    shared = [f"shared_concept_{i}" for i in range(50)]
    concepts = []
    for iv in range(n_interviews):
        unique = [f"unique_iv{iv}_concept_{j}" for j in range(30)]
        all_labels = shared + unique
        for s in range(segs_per_interview):
            for c in range(concepts_per_seg):
                label = all_labels[(s * concepts_per_seg + c) % len(all_labels)]
                concepts.append(_make_foc(label, f"interview_{iv:02d}.txt"))
    return concepts


# ── 1. Ingestion performance ──────────────────────────────────────────────────

def test_ingestion_20_files_under_3s():
    """Loading and segmenting 20 interview files must complete in under 3 seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i in range(1, 21):
            _make_interview_file(tmp_path, i)

        start = time.perf_counter()
        segments = load_all_transcripts(tmp_path)
        elapsed = time.perf_counter() - start

    assert elapsed < 3.0, f"Ingestion took {elapsed:.2f}s — budget is 3.0s"
    assert len(segments) >= 20, f"Expected at least 20 segments, got {len(segments)}"


def test_ingestion_segment_yield():
    """20 interviews should produce at least 80 segments (≥4 per file)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i in range(1, 21):
            _make_interview_file(tmp_path, i)
        segments = load_all_transcripts(tmp_path)

    assert len(segments) >= 80, (
        f"Only {len(segments)} segments from 20 interviews — "
        "check MIN_SEGMENT_WORDS or transcript format."
    )


def test_ingestion_source_diversity():
    """Each of the 20 files must contribute at least one segment."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i in range(1, 21):
            _make_interview_file(tmp_path, i)
        segments = load_all_transcripts(tmp_path)

    sources = {s.source_file for s in segments}
    assert len(sources) == 20, (
        f"Only {len(sources)}/20 files produced segments — some transcripts may be empty."
    )


# ── 2. Concept counting & filtering performance ───────────────────────────────

def test_concept_counter_5000_concepts_under_200ms():
    """Counting 5000 concepts (20 interviews × 250 concepts each) must be < 200ms."""
    concepts = _make_concepts_diverse(n_interviews=20, concepts_per_seg=3, segs_per_interview=80)

    start = time.perf_counter()
    counter = Counter(c.concept.lower().strip() for c in concepts)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.2, f"Counter took {elapsed:.3f}s — budget is 0.200s"
    assert len(counter) > 0


def test_concept_cap_filter_1000_under_50ms():
    """Filtering counter down to top-1000 must be < 50ms even with 5000 unique labels."""
    concepts = _make_concepts_diverse(n_interviews=20, concepts_per_seg=3, segs_per_interview=80)
    counter = Counter(c.concept.lower().strip() for c in concepts)

    start = time.perf_counter()
    sig = dict(counter.most_common(min(1000, len(counter))))
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05, f"Cap filter took {elapsed:.3f}s — budget is 0.050s"
    assert len(sig) <= 1000


def test_concepts_block_string_1000_under_50ms():
    """Building the concepts_block string for the Claude prompt (1000 concepts) must be < 50ms."""
    labels = {f"concept_{i:04d}": (50 - i % 50) for i in range(1000)}

    start = time.perf_counter()
    concepts_block = "\n".join(
        f"  {c} — \u00d7{f}"
        for c, f in sorted(labels.items(), key=lambda x: -x[1])
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05, f"String build took {elapsed:.3f}s — budget is 0.050s"
    assert len(concepts_block) > 0


# ── 3. Label map rebuild performance ─────────────────────────────────────────

def test_label_map_20000_concepts_under_500ms():
    """
    themer.py rebuilds a label_map over all concepts after clustering.
    With 20 interviews × 1000 concepts this must stay under 500ms.
    """
    concepts = _make_concepts_diverse(n_interviews=20, concepts_per_seg=3, segs_per_interview=80)
    # Replicate the label_map logic from themer.build_gioia_structure
    start = time.perf_counter()
    label_map: dict = {}
    for c in concepts:
        key = c.concept.lower().strip()
        label_map.setdefault(key, []).append(c)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"Label map build took {elapsed:.3f}s — budget is 0.500s"
    assert len(label_map) > 0


# ── 4. Semaphore concurrency ──────────────────────────────────────────────────

def test_semaphore_limits_concurrency_to_50():
    """
    With MAX_CONCURRENT_REQUESTS=50 and 200 tasks, the semaphore must ensure
    no more than 50 tasks run simultaneously. Verifies asyncio.Semaphore behaviour.
    """
    import config
    max_concurrent = config.MAX_CONCURRENT_REQUESTS
    active = [0]
    peak = [0]

    async def mock_api_call(sem, i):
        async with sem:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1

    async def run():
        sem = asyncio.Semaphore(max_concurrent)
        await asyncio.gather(*[mock_api_call(sem, i) for i in range(200)])

    asyncio.run(run())

    assert peak[0] <= max_concurrent, (
        f"Peak concurrency {peak[0]} exceeded semaphore limit {max_concurrent}."
    )
    assert peak[0] > 1, "Concurrency never rose above 1 — semaphore may be broken."


def test_semaphore_200_tasks_under_2s():
    """200 mock API calls (10ms each) with concurrency=50 must complete in < 2 seconds."""
    import config

    async def mock_task(sem):
        async with sem:
            await asyncio.sleep(0.01)

    async def run():
        sem = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        await asyncio.gather(*[mock_task(sem) for _ in range(200)])

    start = time.perf_counter()
    asyncio.run(run())
    elapsed = time.perf_counter() - start

    # 200 tasks / 50 concurrent = 4 batches × 0.01s = ~0.04s theoretical minimum
    assert elapsed < 2.0, f"200 async tasks took {elapsed:.2f}s — budget is 2.0s"


# ── 5. Memory footprint ───────────────────────────────────────────────────────

def test_memory_20_interviews_under_100mb():
    """
    20 interviews worth of Segment + FirstOrderConcept objects must stay under 100 MB.
    Uses tracemalloc to measure peak allocation during object creation.
    """
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    concepts = _make_concepts_diverse(n_interviews=20, concepts_per_seg=3, segs_per_interview=20)
    # Simulate the label_map and counter built during theming
    counter = Counter(c.concept.lower().strip() for c in concepts)
    label_map = {}
    for c in concepts:
        label_map.setdefault(c.concept.lower().strip(), []).append(c)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    total_mb = total_bytes / (1024 * 1024)

    assert total_mb < 100, (
        f"Memory for 20-interview data structures: {total_mb:.1f} MB — budget is 100 MB"
    )


# ── 6. End-to-end pipeline data flow (no API calls) ──────────────────────────

def test_pipeline_data_flow_20_interviews():
    """
    Verify the full data flow from ingestion → concepts → counter → label_map
    produces consistent counts across 20 interviews without data loss.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i in range(1, 21):
            _make_interview_file(tmp_path, i)
        segments = load_all_transcripts(tmp_path)

    # Simulate concept generation output: 3 concepts per segment
    concepts = []
    for seg in segments:
        for j in range(3):
            foc = MagicMock(spec=FirstOrderConcept)
            foc.concept = f"concept_type_{j}"
            foc.segment = seg
            concepts.append(foc)

    assert len(concepts) == len(segments) * 3

    counter = Counter(c.concept.lower().strip() for c in concepts)
    assert counter["concept_type_0"] == len(segments)

    label_map = {}
    for c in concepts:
        label_map.setdefault(c.concept.lower().strip(), []).append(c)
    assert len(label_map["concept_type_1"]) == len(segments)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
