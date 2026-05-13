"""
Async segment coding using claude-haiku-4-5 with prompt caching.
The large system prompt is cached after the first request (~90% cost saving
on the system-prompt portion for subsequent calls).
"""
import asyncio
import json
import re
from typing import Callable, List, Optional

import anthropic

from .models import CodedSegment, Segment

_SYSTEM_PROMPT = """You are an expert qualitative researcher conducting inductive thematic analysis \
following Braun & Clarke's (2006) six-phase methodology.

Your task: generate analytical codes for a single interview transcript segment.

Principles:
- Codes are DATA-DRIVEN (bottom-up), not theory-imposed
- Capture semantic content AND latent meaning
- Use active gerund phrases where natural: "seeking belonging", "resisting change"
- Be specific enough to mean something, abstract enough to apply across segments
- Consider what is said AND how (tone, emphasis, hedging)

For each segment return 1–3 concise codes (2–6 words) and a 1-sentence analytical memo.

Return ONLY valid JSON — no markdown, no explanation:
{"codes": ["code one", "code two"], "memo": "One analytical sentence."}"""


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
    return {"codes": ["uncoded"], "memo": ""}


async def _code_one(
    client: anthropic.AsyncAnthropic,
    segment: Segment,
    sem: asyncio.Semaphore,
    model: str,
) -> CodedSegment:
    async with sem:
        label = segment.source_file
        if segment.participant:
            label += f" ({segment.participant})"
        user_msg = f"Segment from {label}:\n\n{segment.text}"
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=350,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            data = _parse(resp.content[0].text)
        except Exception as exc:
            data = {"codes": ["error"], "memo": str(exc)[:120]}
        return CodedSegment(
            segment=segment, codes=data.get("codes", ["uncoded"]), memo=data.get("memo", "")
        )


async def _code_all(
    segments: List[Segment],
    model: str,
    max_concurrent: int,
    on_progress: Optional[Callable[[int, int], None]],
    on_coded: Optional[Callable] = None,
) -> List[CodedSegment]:
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(max_concurrent)
    total = len(segments)
    done = 0
    results: List[Optional[CodedSegment]] = [None] * total

    async def _wrap(idx: int, seg: Segment) -> None:
        nonlocal done
        cs = await _code_one(client, seg, sem, model)
        results[idx] = cs
        done += 1
        if on_coded:
            on_coded(cs)
        if on_progress:
            on_progress(done, total)

    await asyncio.gather(*[_wrap(i, s) for i, s in enumerate(segments)])
    return [r for r in results if r is not None]


def code_segments(
    segments: List[Segment],
    model: str = "claude-haiku-4-5",
    max_concurrent: int = 15,
    on_progress: Optional[Callable[[int, int], None]] = None,
    on_coded: Optional[Callable] = None,
) -> List[CodedSegment]:
    """Synchronous entry point; runs async coding internally."""
    return asyncio.run(_code_all(segments, model, max_concurrent, on_progress, on_coded))
