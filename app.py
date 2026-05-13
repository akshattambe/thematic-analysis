"""
Web UI server for the Thematic Analysis Tool.
Usage: python app.py   then open http://localhost:8000
"""
import asyncio
import json
import shutil
import tempfile
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

import config

# Braun & Clarke imports
from src.coder import code_segments
from src.ingestion import load_all_transcripts
from src.models import AnalysisResult
from src.nlp import run_nlp
from src.reporter import generate_html as bc_generate_html
from src.reporter import save_excel as bc_save_excel
from src.reporter import save_json as bc_save_json
from src.themer import build_themes
from src.visualizer import generate_all as bc_generate_all

# Gioia imports
from src_gioia.coder import code_segments_gioia
from src_gioia.models import GioiaResult
from src_gioia.reporter import generate_html as gioia_generate_html
from src_gioia.reporter import save_excel as gioia_save_excel
from src_gioia.reporter import save_json as gioia_save_json
from src_gioia.themer import build_gioia_structure
from src_gioia.visualizer import generate_all_gioia

app = FastAPI(title="Thematic Analysis")

# In-memory job store: job_id -> {status, events, done, out_dir}
jobs: Dict[str, dict] = {}


# ── Braun & Clarke pipeline ───────────────────────────────────────────────────

def _run_bc_pipeline(job_id: str, tmp_dir: Path, out_dir: Path) -> None:
    job = jobs[job_id]

    def emit(event: dict) -> None:
        job["events"].append(event)

    try:
        emit({"type": "stage", "stage": "ingesting", "message": "Loading transcripts…"})
        segments = load_all_transcripts(tmp_dir)
        if not segments:
            emit({"type": "error", "message": "No segments found — check transcript format."})
            return
        n_files = len({s.source_file for s in segments})
        emit({"type": "ingested", "segment_count": len(segments), "file_count": n_files})

        emit({"type": "stage", "stage": "coding", "message": "Generating codes with AI…"})
        coded_count = [0]

        def on_coded(cs) -> None:
            coded_count[0] += 1
            emit({"type": "coded_segment", "codes": cs.codes,
                  "done": coded_count[0], "total": len(segments)})

        coded_segs = code_segments(
            segments, model=config.CODING_MODEL,
            max_concurrent=config.MAX_CONCURRENT_REQUESTS, on_coded=on_coded,
        )

        code_freq: Counter = Counter()
        for cs in coded_segs:
            for c in cs.codes:
                code_freq[c] += 1
        emit({"type": "coded", "total_codes": sum(code_freq.values()),
              "unique_codes": len(code_freq)})

        emit({"type": "stage", "stage": "theming", "message": "Synthesising themes with AI…"})

        def on_theme_ready(theme) -> None:
            emit({"type": "theme_ready", "theme": {
                "id": theme.id, "name": theme.name, "description": theme.description,
                "codes": theme.codes, "segment_count": theme.segment_count,
                "representative_quotes": [
                    {"text": q.text, "source_file": q.source_file, "participant": q.participant}
                    for q in theme.representative_quotes[:3]
                ],
            }})

        themes = build_themes(
            coded_segs, min_themes=4, max_themes=10,
            on_progress=lambda msg: emit({"type": "theme_progress", "message": msg}),
            on_theme_ready=on_theme_ready,
        )

        emit({"type": "stage", "stage": "reporting", "message": "Generating reports…"})
        nlp_data = run_nlp(segments, coded_segs)
        result = AnalysisResult(
            timestamp=datetime.now().isoformat(),
            source_files=sorted({s.source_file for s in segments}),
            total_segments=len(segments), coded_segments=coded_segs,
            themes=themes, nlp_data=nlp_data,
        )
        viz = bc_generate_all(result)
        bc_generate_html(result, viz, out_dir, config.TEMPLATES_DIR)
        bc_save_excel(result, out_dir)
        bc_save_json(result, out_dir)

        emit({
            "type": "complete",
            "methodology": "braun_clarke",
            "stats": {"total_themes": len(themes), "total_segments": len(segments),
                      "total_files": n_files, "total_codes": len(code_freq)},
            "themes": [
                {"id": t.id, "name": t.name, "description": t.description,
                 "codes": t.codes, "segment_count": t.segment_count,
                 "representative_quotes": [
                     {"text": q.text, "source_file": q.source_file, "participant": q.participant}
                     for q in t.representative_quotes[:3]
                 ]}
                for t in themes
            ],
            "code_frequencies": dict(code_freq.most_common(50)),
        })
        job["status"] = "complete"

    except Exception as exc:
        import traceback
        emit({"type": "error", "message": str(exc), "detail": traceback.format_exc()})
        job["status"] = "error"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        job["done"] = True


# ── Gioia pipeline ────────────────────────────────────────────────────────────

def _run_gioia_pipeline(job_id: str, tmp_dir: Path, out_dir: Path) -> None:
    job = jobs[job_id]

    def emit(event: dict) -> None:
        job["events"].append(event)

    try:
        emit({"type": "stage", "stage": "ingesting", "message": "Loading transcripts…"})
        segments = load_all_transcripts(tmp_dir)
        if not segments:
            emit({"type": "error", "message": "No segments found — check transcript format."})
            return
        n_files = len({s.source_file for s in segments})
        emit({"type": "ingested", "segment_count": len(segments), "file_count": n_files})

        emit({"type": "stage", "stage": "coding",
              "message": "Generating 1st-order concepts with AI…"})
        coded_count = [0]

        def on_coded_gioia(concepts_list) -> None:
            coded_count[0] += 1
            emit({"type": "coded_segment",
                  "concepts": [{"concept": c.concept, "verbatim": c.verbatim[:70]}
                                for c in concepts_list],
                  "done": coded_count[0], "total": len(segments)})

        concepts = code_segments_gioia(
            segments, model=config.CODING_MODEL,
            max_concurrent=config.MAX_CONCURRENT_REQUESTS, on_coded=on_coded_gioia,
        )
        emit({"type": "coded", "total_codes": len(concepts),
              "unique_codes": len({c.concept for c in concepts})})

        emit({"type": "stage", "stage": "theming",
              "message": "Building 2nd-order themes and aggregate dimensions…"})

        def on_theme_ready(theme) -> None:
            emit({"type": "theme_ready",
                  "theme": {"id": theme.id, "name": theme.name}})

        def on_dimension_ready(dim) -> None:
            emit({"type": "dim_ready",
                  "dimension": {"id": dim.id, "name": dim.name,
                                "theme_count": dim.theme_count}})

        second_order, aggregate = build_gioia_structure(
            concepts, min_themes=5, max_themes=25,
            on_progress=lambda msg: emit({"type": "theme_progress", "message": msg}),
            on_theme_ready=on_theme_ready,
            on_dimension_ready=on_dimension_ready,
        )

        emit({"type": "stage", "stage": "reporting", "message": "Generating reports…"})
        nlp_data = run_nlp(segments)
        result = GioiaResult(
            timestamp=datetime.now().isoformat(),
            source_files=sorted({s.source_file for s in segments}),
            total_segments=len(segments),
            first_order_concepts=concepts,
            second_order_themes=second_order,
            aggregate_dimensions=aggregate,
            nlp_data=nlp_data,
        )
        viz = generate_all_gioia(result)
        gioia_generate_html(result, viz, out_dir, config.TEMPLATES_DIR)
        gioia_save_excel(result, out_dir)
        gioia_save_json(result, out_dir)

        emit({
            "type": "complete",
            "methodology": "gioia",
            "stats": {
                "total_dimensions": len(aggregate),
                "total_second_order": len(second_order),
                "total_first_order": len(concepts),
                "total_segments": len(segments),
                "total_files": n_files,
            },
            "dimensions": [
                {
                    "id": d.id, "name": d.name, "description": d.description,
                    "second_order_themes": [
                        {
                            "id": t.id, "name": t.name, "description": t.description,
                            "concept_count": t.concept_count,
                            "first_order_concepts": [
                                {"concept": c.concept, "verbatim": c.verbatim,
                                 "source_file": c.segment.source_file,
                                 "participant": c.segment.participant}
                                for c in t.first_order_concepts[:8]
                            ],
                        }
                        for t in d.second_order_themes
                    ],
                }
                for d in aggregate
            ],
        })
        job["status"] = "complete"

    except Exception as exc:
        import traceback
        emit({"type": "error", "message": str(exc), "detail": traceback.format_exc()})
        job["status"] = "error"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        job["done"] = True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((config.TEMPLATES_DIR / "app.html").read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def start_analysis(
    files: List[UploadFile] = File(...),
    methodology: str = Form("gioia"),
):
    if not files:
        raise HTTPException(400, "No files provided")

    job_id = str(uuid.uuid4())[:8]
    tmp_dir = Path(tempfile.mkdtemp())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.OUTPUTS_DIR / f"analysis_{ts}_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        fname = f.filename or f"file_{uuid.uuid4().hex[:6]}"
        (tmp_dir / fname).write_bytes(await f.read())

    jobs[job_id] = {"status": "running", "events": [], "done": False, "out_dir": str(out_dir)}

    pipeline = _run_gioia_pipeline if methodology == "gioia" else _run_bc_pipeline
    threading.Thread(target=pipeline, args=(job_id, tmp_dir, out_dir), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream_events(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def generator():
        idx = 0
        while True:
            job = jobs[job_id]
            while idx < len(job["events"]):
                yield f"data: {json.dumps(job['events'][idx])}\n\n"
                idx += 1
            if job["done"]:
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/report/{job_id}", response_class=HTMLResponse)
async def get_report(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    out_dir = Path(jobs[job_id]["out_dir"])
    # Try Gioia report first, then BC report
    for name in ("gioia_report.html", "report.html"):
        path = out_dir / name
        if path.exists():
            return HTMLResponse(path.read_text(encoding="utf-8"))
    raise HTTPException(404, "Report not ready")


@app.get("/api/download/{job_id}/excel")
async def download_excel(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    out_dir = Path(jobs[job_id]["out_dir"])
    for name in ("gioia_results.xlsx", "analysis_results.xlsx"):
        path = out_dir / name
        if path.exists():
            return FileResponse(
                path, filename=name,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    raise HTTPException(404, "Excel not ready")


@app.get("/api/download/{job_id}/json")
async def download_json(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    out_dir = Path(jobs[job_id]["out_dir"])
    for name in ("gioia_results.json", "analysis_results.json"):
        path = out_dir / name
        if path.exists():
            return FileResponse(path, filename=name, media_type="application/json")
    raise HTTPException(404, "JSON not ready")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
