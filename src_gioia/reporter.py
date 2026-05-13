import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .models import GioiaResult


def save_json(result: GioiaResult, out_dir: Path) -> Path:
    data = {
        "timestamp": result.timestamp,
        "source_files": result.source_files,
        "total_segments": result.total_segments,
        "aggregate_dimensions": [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "second_order_themes": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "description": t.description,
                        "first_order_concepts": [
                            {
                                "id": c.id,
                                "concept": c.concept,
                                "verbatim": c.verbatim,
                                "source_file": c.segment.source_file,
                                "participant": c.segment.participant,
                                "memo": c.memo,
                            }
                            for c in t.first_order_concepts
                        ],
                    }
                    for t in d.second_order_themes
                ],
            }
            for d in result.aggregate_dimensions
        ],
        "nlp_data": result.nlp_data,
    }
    path = out_dir / "gioia_results.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def save_excel(result: GioiaResult, out_dir: Path) -> Path:
    path = out_dir / "gioia_results.xlsx"

    dim_of = {
        c.id: d.name
        for d in result.aggregate_dimensions
        for t in d.second_order_themes
        for c in t.first_order_concepts
    }
    theme_of = {
        c.id: t.name
        for t in result.second_order_themes
        for c in t.first_order_concepts
    }

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([
            {
                "Dimension ID": d.id,
                "Dimension Name": d.name,
                "Description": d.description,
                "2nd-Order Theme Count": d.theme_count,
                "1st-Order Concept Count": d.concept_count,
            }
            for d in result.aggregate_dimensions
        ]).to_excel(writer, sheet_name="Aggregate Dimensions", index=False)

        pd.DataFrame([
            {
                "Theme ID": t.id,
                "Theme Name": t.name,
                "Description": t.description,
                "Aggregate Dimension": next(
                    (d.name for d in result.aggregate_dimensions
                     if any(x.id == t.id for x in d.second_order_themes)), ""
                ),
                "Concept Count": t.concept_count,
            }
            for t in result.second_order_themes
        ]).to_excel(writer, sheet_name="2nd Order Themes", index=False)

        pd.DataFrame([
            {
                "Concept ID": c.id,
                "1st-Order Concept": c.concept,
                "Verbatim Quote": c.verbatim,
                "2nd-Order Theme": theme_of.get(c.id, ""),
                "Aggregate Dimension": dim_of.get(c.id, ""),
                "Source File": c.segment.source_file,
                "Participant": c.segment.participant or "",
                "Segment Text": c.segment.text[:300],
                "Memo": c.memo,
            }
            for c in result.first_order_concepts
        ]).to_excel(writer, sheet_name="1st Order Concepts", index=False)

    return path


def generate_html(result: GioiaResult, viz: Dict, out_dir: Path, templates_dir: Path) -> Path:
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)
    template = env.get_template("gioia_report.html")
    html = template.render(
        result=result,
        viz=viz,
        generated_at=datetime.now().strftime("%B %d, %Y at %H:%M"),
        total_dimensions=len(result.aggregate_dimensions),
        total_second_order=len(result.second_order_themes),
        total_first_order=len(result.first_order_concepts),
        total_segments=result.total_segments,
        total_files=len(result.source_files),
    )
    path = out_dir / "gioia_report.html"
    path.write_text(html, encoding="utf-8")
    return path
