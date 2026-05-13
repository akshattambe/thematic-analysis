#!/usr/bin/env python3
"""
Thematic Analysis CLI
Usage:
  python main.py analyze                  # run full pipeline on data/transcripts/
  python main.py analyze -t /path/to/dir  # custom transcripts directory
  python main.py report data/outputs/analysis_YYYYMMDD_HHMMSS
                                          # regenerate report from saved results
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

import config
from src.ingestion import load_all_transcripts
from src.models import AnalysisResult, CodedSegment, Quote, Segment, Theme
from src.nlp import run_nlp
from src.coder import code_segments
from src.themer import build_themes
from src.visualizer import generate_all
from src.reporter import generate_html, save_excel, save_json

console = Console()


# ── helpers ──────────────────────────────────────────────────────────────────

def _seg_from_dict(d: dict) -> Segment:
    return Segment(
        id=d["id"],
        source_file=d["source_file"],
        participant=d.get("participant"),
        text=d["text"],
        position=d.get("position", 0),
        word_count=d.get("word_count", len(d["text"].split())),
    )


def _load_coded_cache(path: Path, seg_map: dict) -> List[CodedSegment]:
    data = json.loads(path.read_text())
    out: List[CodedSegment] = []
    for d in data:
        seg = seg_map.get(d["id"])
        if seg:
            out.append(CodedSegment(segment=seg, codes=d["codes"], memo=d.get("memo", "")))
    return out


def _load_themes_cache(path: Path, coded_segs: List[CodedSegment]) -> List[Theme]:
    data = json.loads(path.read_text())
    themes: List[Theme] = []
    for td in data:
        codes_set = set(td["codes"])
        seg_list = [cs for cs in coded_segs if any(c in codes_set for c in cs.codes)]
        quotes = [Quote(**q) for q in td.get("representative_quotes", [])]
        themes.append(
            Theme(
                id=td["id"],
                name=td["name"],
                description=td["description"],
                codes=td["codes"],
                representative_quotes=quotes,
                coded_segments=seg_list,
            )
        )
    return themes


# ── commands ─────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Thematic Analysis Tool · Powered by Claude"""


@cli.command()
@click.option("--transcripts", "-t", default=str(config.TRANSCRIPTS_DIR),
              type=click.Path(), show_default=True,
              help="Directory containing .txt / .docx / .pdf transcript files")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output directory (default: data/outputs/analysis_TIMESTAMP)")
@click.option("--min-themes", default=4, show_default=True, type=int)
@click.option("--max-themes", default=10, show_default=True, type=int)
@click.option("--coding-model", default=config.CODING_MODEL, show_default=True,
              help="Model for segment coding")
@click.option("--force", is_flag=True, help="Re-run all steps, ignoring cache")
@click.option("--no-viz", is_flag=True, help="Skip visualisation generation")
def analyze(transcripts, output, min_themes, max_themes, coding_model, force, no_viz):
    """Run full thematic analysis pipeline."""

    transcripts_dir = Path(transcripts)
    if not transcripts_dir.exists():
        console.print(f"[red]Transcripts directory not found: {transcripts_dir}[/red]")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output) if output else config.OUTPUTS_DIR / f"analysis_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"Transcripts : {transcripts_dir}\n"
            f"Output      : {out_dir}\n"
            f"Coding model: {coding_model}\n"
            f"Synthesis   : claude-opus-4-7 (adaptive thinking)",
            title="[bold]Thematic Analysis Tool[/bold]",
            border_style="blue",
        )
    )

    # ── Step 1: Ingest ────────────────────────────────────────────────────
    console.rule("[bold blue]Step 1 · Loading transcripts")
    segments = load_all_transcripts(transcripts_dir)
    if not segments:
        console.print("[red]No segments found. Add .txt/.docx/.pdf files to the transcripts folder.[/red]")
        sys.exit(1)

    n_files = len({s.source_file for s in segments})
    console.print(f"[green]✓[/green] {len(segments)} segments from {n_files} file(s)")

    # Persist segments for resuming
    seg_cache = out_dir / "segments.json"
    if force or not seg_cache.exists():
        seg_cache.write_text(
            json.dumps(
                [{"id": s.id, "source_file": s.source_file, "participant": s.participant,
                  "text": s.text, "position": s.position, "word_count": s.word_count}
                 for s in segments],
                ensure_ascii=False,
            )
        )

    seg_map = {s.id: s for s in segments}

    # ── Step 2: Code ──────────────────────────────────────────────────────
    console.rule("[bold blue]Step 2 · Coding segments")
    coded_cache = out_dir / "coded_segments.json"

    if not force and coded_cache.exists():
        console.print("[dim]Resuming from cached coded segments…[/dim]")
        coded_segs = _load_coded_cache(coded_cache, seg_map)
    else:
        coded_segs: List[CodedSegment] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task("Coding segments…", total=len(segments), completed=0)

            def _on_progress(done: int, total: int):
                prog.update(task, completed=done, total=total)

            coded_segs = code_segments(
                segments,
                model=coding_model,
                max_concurrent=config.MAX_CONCURRENT_REQUESTS,
                on_progress=_on_progress,
            )

        coded_cache.write_text(
            json.dumps(
                [{"id": cs.segment.id, "codes": cs.codes, "memo": cs.memo}
                 for cs in coded_segs],
                ensure_ascii=False,
            )
        )

    n_codes = sum(len(cs.codes) for cs in coded_segs)
    console.print(f"[green]✓[/green] {n_codes} codes across {len(coded_segs)} segments")

    # NLP enrichment
    nlp_data = run_nlp(segments, coded_segs)

    # ── Step 3: Themes ────────────────────────────────────────────────────
    console.rule("[bold blue]Step 3 · Synthesising themes")
    themes_cache = out_dir / "themes.json"

    if not force and themes_cache.exists():
        console.print("[dim]Resuming from cached themes…[/dim]")
        themes = _load_themes_cache(themes_cache, coded_segs)
    else:
        def _log(msg: str):
            console.print(f"  [dim]{msg}[/dim]")

        themes = build_themes(
            coded_segs,
            min_themes=min_themes,
            max_themes=max_themes,
            on_progress=_log,
        )
        themes_cache.write_text(
            json.dumps(
                [
                    {
                        "id": t.id,
                        "name": t.name,
                        "description": t.description,
                        "codes": t.codes,
                        "representative_quotes": [
                            {"text": q.text, "source_file": q.source_file, "participant": q.participant}
                            for q in t.representative_quotes
                        ],
                    }
                    for t in themes
                ],
                ensure_ascii=False,
            )
        )

    console.print(f"[green]✓[/green] {len(themes)} themes identified")

    # ── Step 4: Reports ───────────────────────────────────────────────────
    console.rule("[bold blue]Step 4 · Generating reports")

    result = AnalysisResult(
        timestamp=datetime.now().isoformat(),
        source_files=sorted({s.source_file for s in segments}),
        total_segments=len(segments),
        coded_segments=coded_segs,
        themes=themes,
        nlp_data=nlp_data,
    )

    viz = {}
    if not no_viz:
        with console.status("Building visualisations…"):
            viz = generate_all(result)
        console.print("[green]✓[/green] Visualisations ready")

    json_path = save_json(result, out_dir)
    excel_path = save_excel(result, out_dir)
    html_path = generate_html(result, viz, out_dir, config.TEMPLATES_DIR)

    console.print(f"[green]✓[/green] JSON   → {json_path.name}")
    console.print(f"[green]✓[/green] Excel  → {excel_path.name}")
    console.print(f"[green]✓[/green] Report → {html_path}")

    # Summary table
    tbl = Table(title="\nTheme Summary", show_header=True, header_style="bold")
    tbl.add_column("#", style="dim", width=3)
    tbl.add_column("Theme", min_width=30)
    tbl.add_column("Codes", justify="right", width=6)
    tbl.add_column("Segments", justify="right", width=9)
    for t in themes:
        tbl.add_row(str(t.id), t.name, str(len(t.codes)), str(t.segment_count))
    console.print(tbl)

    console.print(f"\n[bold green]Done! Open the report:[/bold green] {html_path}")


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
def report(results_dir):
    """Regenerate HTML + Excel from a previous analysis (results_dir must contain analysis_results.json)."""
    out_dir = Path(results_dir)
    json_path = out_dir / "analysis_results.json"
    if not json_path.exists():
        console.print(f"[red]analysis_results.json not found in {out_dir}[/red]")
        sys.exit(1)

    data = json.loads(json_path.read_text())
    console.print(f"[green]Loaded analysis from[/green] {json_path}")

    coded_segs: List[CodedSegment] = []
    for d in data.get("coded_segments", []):
        seg = _seg_from_dict(d)
        coded_segs.append(CodedSegment(segment=seg, codes=d["codes"], memo=d.get("memo", "")))

    themes: List[Theme] = _load_themes_cache(
        out_dir / "themes.json" if (out_dir / "themes.json").exists() else json_path,
        coded_segs,
    )

    result = AnalysisResult(
        timestamp=data.get("timestamp", ""),
        source_files=data.get("source_files", []),
        total_segments=data.get("total_segments", len(coded_segs)),
        coded_segments=coded_segs,
        themes=themes,
        nlp_data=data.get("nlp_data", {}),
    )

    with console.status("Building visualisations…"):
        viz = generate_all(result)

    html_path = generate_html(result, viz, out_dir, config.TEMPLATES_DIR)
    excel_path = save_excel(result, out_dir)
    console.print(f"[green]✓[/green] {html_path}")
    console.print(f"[green]✓[/green] {excel_path}")


if __name__ == "__main__":
    cli()
