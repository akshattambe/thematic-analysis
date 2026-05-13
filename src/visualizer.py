"""Visualizations: plotly charts (returned as HTML) + matplotlib word clouds (base64 PNG)."""
import base64
from collections import Counter
from io import BytesIO
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud

from .models import AnalysisResult, CodedSegment, Theme

_PALETTE = [
    "#2E86AB","#A23B72","#F18F01","#C73E1D","#3B6064",
    "#44BBA4","#E94F37","#7B68EE","#20B2AA","#FF6B6B",
    "#4ECDC4","#45B7D1","#96CEB4","#FFEAA7","#DDA0DD",
]


def _to_html(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def theme_frequency_chart(themes: List[Theme]) -> str:
    names = [t.name for t in themes]
    counts = [t.segment_count for t in themes]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(themes))]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color=colors,
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Theme Frequency (Segments)",
        xaxis_title="Segments",
        height=max(300, len(themes) * 52 + 100),
        margin=dict(l=20, r=50, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia,serif", size=13),
        yaxis=dict(autorange="reversed"),
    )
    return _to_html(fig)


def code_frequency_chart(coded_segments: List[CodedSegment], top_n: int = 25) -> str:
    counter: Counter = Counter()
    for cs in coded_segments:
        for c in cs.codes:
            counter[c] += 1

    top = counter.most_common(top_n)
    codes = [t[0] for t in top]
    freqs = [t[1] for t in top]

    fig = go.Figure(
        go.Bar(
            x=freqs,
            y=codes,
            orientation="h",
            marker_color="#2E86AB",
            text=freqs,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Code Frequencies",
        xaxis_title="Frequency",
        height=max(400, top_n * 24 + 100),
        margin=dict(l=20, r=50, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia,serif", size=11),
        yaxis=dict(autorange="reversed"),
    )
    return _to_html(fig)


def source_theme_heatmap(themes: List[Theme]) -> str:
    sources = sorted({cs.segment.source_file for t in themes for cs in t.coded_segments})
    if not sources:
        return ""

    matrix = []
    for t in themes:
        counts = Counter(cs.segment.source_file for cs in t.coded_segments)
        matrix.append([counts.get(s, 0) for s in sources])

    short_src = [s[:18] + "…" if len(s) > 18 else s for s in sources]
    short_thm = [t.name[:28] + "…" if len(t.name) > 28 else t.name for t in themes]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=short_src,
            y=short_thm,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Theme Distribution Across Transcripts",
        height=max(300, len(themes) * 42 + 150),
        xaxis=dict(tickangle=-40),
        margin=dict(l=20, r=20, t=60, b=100),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia,serif", size=11),
    )
    return _to_html(fig)


def word_frequency_chart(freq: Dict[str, int], top_n: int = 30) -> str:
    items = sorted(freq.items(), key=lambda x: -x[1])[:top_n]
    words = [w for w, _ in items]
    counts = [c for _, c in items]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=words,
            orientation="h",
            marker_color="#44BBA4",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Word Frequencies",
        xaxis_title="Frequency",
        height=max(400, top_n * 22 + 100),
        margin=dict(l=20, r=40, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia,serif", size=11),
        yaxis=dict(autorange="reversed"),
    )
    return _to_html(fig)


def theme_wordcloud(theme: Theme) -> str:
    """Return a data-URI PNG word cloud for a theme's codes + quote words."""
    # Codes weighted 3×, representative quote text weighted 1×
    tokens = []
    for code in theme.codes:
        tokens.extend([code.replace(" ", "_")] * 3)
    for q in theme.representative_quotes:
        tokens.extend(q.text.split())

    text = " ".join(tokens) if tokens else theme.name

    try:
        wc = WordCloud(
            width=900, height=420,
            background_color="white",
            colormap="viridis",
            max_words=60,
            min_font_size=10,
            max_font_size=90,
            collocations=False,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(theme.name, fontsize=14, fontweight="bold", pad=10)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def generate_all(result: AnalysisResult) -> Dict:
    viz: Dict = {}

    if result.themes:
        viz["theme_frequency"] = theme_frequency_chart(result.themes)
        hm = source_theme_heatmap(result.themes)
        if hm:
            viz["heatmap"] = hm
        viz["wordclouds"] = {}
        for t in result.themes:
            wc = theme_wordcloud(t)
            if wc:
                viz["wordclouds"][t.id] = wc

    if result.coded_segments:
        viz["code_frequency"] = code_frequency_chart(result.coded_segments)

    if result.nlp_data.get("word_frequencies"):
        viz["word_frequency"] = word_frequency_chart(result.nlp_data["word_frequencies"])

    return viz
