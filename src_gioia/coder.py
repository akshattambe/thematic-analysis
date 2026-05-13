"""
Async first-order concept generation using claude-haiku-4-5 with prompt caching.
Gioia Methodology: labels stay close to informant language with verbatim excerpts.
"""
import asyncio
import json
import re
from typing import Callable, List, Optional

import anthropic

from src.models import Segment
from .models import FirstOrderConcept

_SYSTEM_PROMPT = """You are conducting rigorous qualitative research using Gioia Methodology \
(Gioia, Corley & Hamilton, 2013).

Your task: generate FIRST-ORDER CONCEPTS from a single interview transcript segment.

First-order concepts must:
- Stay close to the DATA and the INFORMANT'S OWN LANGUAGE and vocabulary
- Use the participant's actual words and phrases where possible
- Preserve the informant's perspective (descriptive, NOT researcher interpretation)
- Each concept requires a VERBATIM EXCERPT: an exact direct quote of 8–20 words from the segment

Generate 2–4 first-order concepts per segment.

Return ONLY valid JSON — no markdown, no explanation:
{"concepts": [{"concept": "label in informant's language", "verbatim": "exact quote from the text"}], \
"memo": "One analytical sentence about this segment."}"""


def _parse(text: str) -> dict:
    text = text.strip()
    for wrapper in ["```json", "```"]:
        if wrapper in text:
            text = text.split(wrapper, 1)[1].split("```")[0]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"concepts": [{"concept": "uncoded", "verbatim": ""}], "memo": ""}


async def _code_one(
    client: anthropic.AsyncAnthropic,
    segment: Segment,
    sem: asyncio.Semaphore,
    model: str,
) -> List[FirstOrderConcept]:
    async with sem:
        label = segment.source_file
        if segment.participant:
            label += f" ({segment.participant})"
        user_msg = f"Segment from {label}:\n\n{segment.text}"
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=500,
                system=[{"type": "text", "text": _SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_msg}],
            )
            data = _parse(resp.content[0].text)
        except Exception as exc:
            data = {"concepts": [{"concept": "error", "verbatim": ""}], "memo": str(exc)[:120]}

        memo = data.get("memo", "")
        concepts: List[FirstOrderConcept] = []
        for c in data.get("concepts", [])[:4]:
            if isinstance(c, dict) and c.get("concept"):
                concepts.append(FirstOrderConcept(
                    segment=segment,
                    concept=c["concept"],
                    verbatim=c.get("verbatim", ""),
                    memo=memo,
                ))
        if not concepts:
            concepts.append(FirstOrderConcept(segment=segment, concept="uncoded",
                                               verbatim="", memo=memo))
        return concepts


async def _code_all(
    segments: List[Segment],
    model: str,
    max_concurrent: int,
    on_progress: Optional[Callable],
    on_coded: Optional[Callable],
) -> List[FirstOrderConcept]:
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(max_concurrent)
    total = len(segments)
    done = 0
    all_concepts: List[FirstOrderConcept] = []

    async def _wrap(seg: Segment) -> None:
        nonlocal done
        concepts = await _code_one(client, seg, sem, model)
        all_concepts.extend(concepts)
        done += 1
        if on_coded:
            on_coded(concepts)
        if on_progress:
            on_progress(done, total)

    await asyncio.gather(*[_wrap(s) for s in segments])
    return all_concepts


def code_segments_gioia(
    segments: List[Segment],
    model: str = "claude-haiku-4-5",
    max_concurrent: int = 15,
    on_progress: Optional[Callable] = None,
    on_coded: Optional[Callable] = None,
) -> List[FirstOrderConcept]:
    """Generate first-order concepts for all segments. Synchronous entry point."""
    return asyncio.run(_code_all(segments, model, max_concurrent, on_progress, on_coded))
