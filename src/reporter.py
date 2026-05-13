import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .models import AnalysisResult


def save_json(result: AnalysisResult, out_dir: Path) -> Path:
    data = {
        "timestamp": result.timestamp,
        "source_files": result.source_files,
        "total_segments": result.total_segments,
        "themes": [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "codes": t.codes,
                "segment_count": t.segment_count,
                "representative_quotes": [
                    {"text": q.text, "source_file": q.source_file, "participant": q.participant}
                    for q in t.representative_quotes
                ],
            }
            for t in result.themes
        ],
        "coded_segments": [
            {
                "id": cs.segment.id,
                "source_file": cs.segment.source_file,
                "participant": cs.segment.participant,
                "text": cs.segment.text,
                "word_count": cs.segment.word_count,
                "codes": cs.codes,
                "memo": cs.memo,
            }
            for cs in result.coded_segments
        ],
        "nlp_data": result.nlp_data,
    }
    path = out_dir / "analysis_results.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def save_excel(result: AnalysisResult, out_dir: Path) -> Path:
    path = out_dir / "analysis_results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Themes
        pd.DataFrame(
            [
                {
                    "Theme ID": t.id,
                    "Theme Name": t.name,
                    "Description": t.description,
                    "Codes": ", ".join(t.codes),
                    "Segment Count": t.segment_count,
                    "Key Quote": t.representative_quotes[0].text if t.representative_quotes else "",
                }
                for t in result.themes
            ]
        ).to_excel(writer, sheet_name="Themes", index=False)

        # Coded segments
        pd.DataFrame(
            [
                {
                    "Segment ID": cs.segment.id,
                    "Source File": cs.segment.source_file,
                    "Participant": cs.segment.participant or "",
                    "Text": cs.segment.text,
                    "Word Count": cs.segment.word_count,
                    "Codes": ", ".join(cs.codes),
                    "Memo": cs.memo,
                }
                for cs in result.coded_segments
            ]
        ).to_excel(writer, sheet_name="Coded Segments", index=False)

        # Code frequencies
        freq = result.nlp_data.get("code_frequencies", {})
        if freq:
            pd.DataFrame(
                [{"Code": c, "Frequency": f} for c, f in freq.items()]
            ).to_excel(writer, sheet_name="Code Frequencies", index=False)

        # Representative quotes
        quotes_rows = [
            {
                "Theme": t.name,
                "Quote": q.text,
                "Source File": q.source_file,
                "Participant": q.participant or "",
            }
            for t in result.themes
            for q in t.representative_quotes
        ]
        if quotes_rows:
            pd.DataFrame(quotes_rows).to_excel(writer, sheet_name="Key Quotes", index=False)

    return path


def generate_html(
    result: AnalysisResult,
    visualizations: Dict,
    out_dir: Path,
    templates_dir: Path,
) -> Path:
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)
    template = env.get_template("report.html")

    html = template.render(
        result=result,
        themes=result.themes,
        viz=visualizations,
        generated_at=datetime.now().strftime("%B %d, %Y at %H:%M"),
        total_themes=len(result.themes),
        total_files=len(result.source_files),
        total_segments=result.total_segments,
        total_codes=len(result.nlp_data.get("code_frequencies", {})),
    )

    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
