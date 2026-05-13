"""Visualizations for Gioia methodology: data structure table, sunburst, concept cloud."""
import base64
from collections import Counter
from io import BytesIO
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud

from .models import AggregateDimension, FirstOrderConcept, GioiaResult

_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B6064",
    "#44BBA4", "#E94F37", "#7B68EE", "#20B2AA", "#FF6B6B",
]


def _to_html(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def data_structure_table(dimensions: List[AggregateDimension]) -> str:
    """Generate the classic Gioia 3-column data structure table as HTML."""
    rows: List[str] = []

    for dim_idx, dim in enumerate(dimensions):
        color = _PALETTE[dim_idx % len(_PALETTE)]
        dim_concept_count = dim.concept_count or 1
        first_dim = True

        for theme in dim.second_order_themes:
            theme_concept_count = theme.concept_count or 1
            first_theme = True

            for foc in theme.first_order_concepts:
                cells: List[str] = []

                # Column 1: 1st-order concept + verbatim
                vb = foc.verbatim[:90] + ("…" if len(foc.verbatim) > 90 else "")
                cells.append(
                    f'<td class="ds-foc">'
                    f'<span class="ds-concept-label">{_esc(foc.concept)}</span>'
                    + (f'<span class="ds-verbatim">"{_esc(vb)}"</span>' if vb else "")
                    + "</td>"
                )

                # Column 2: 2nd-order theme (rowspan = concept count for this theme)
                if first_theme:
                    cells.append(
                        f'<td class="ds-theme" rowspan="{theme_concept_count}">'
                        f"{_esc(theme.name)}</td>"
                    )
                    first_theme = False

                # Column 3: aggregate dimension (rowspan = concept count for this dimension)
                if first_dim:
                    cells.append(
                        f'<td class="ds-dim" rowspan="{dim_concept_count}" '
                        f'style="border-left:4px solid {color};">'
                        f'<strong style="color:{color}">{_esc(dim.name)}</strong>'
                        f"</td>"
                    )
                    first_dim = False

                rows.append(f'<tr>{"".join(cells)}</tr>')

    return (
        '<div class="ds-scroll">'
        '<table class="ds-table">'
        "<thead><tr>"
        "<th>1st Order Concepts</th>"
        "<th>2nd Order Themes</th>"
        "<th>Aggregate Dimensions</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table></div>"
    )


def hierarchy_sunburst(dimensions: List[AggregateDimension]) -> str:
    """Plotly sunburst: inner ring = dimensions, outer ring = 2nd-order themes."""
    # Use branchvalues="remainder": each node's sector size = own value + children's values.
    # Dimension nodes get value=0 so their size is derived entirely from their theme children.
    ids     = ["root"]
    labels  = [""]
    parents = [""]
    values  = [0]
    colors  = ["rgba(0,0,0,0)"]

    for i, dim in enumerate(dimensions):
        color  = _PALETTE[i % len(_PALETTE)]
        dim_id = f"dim_{dim.id}"
        ids.append(dim_id);    labels.append(dim.name)
        parents.append("root"); values.append(0); colors.append(color)

        for theme in dim.second_order_themes:
            t_id = f"t_{theme.id}"
            ids.append(t_id);         labels.append(theme.name)
            parents.append(dim_id);   values.append(max(theme.concept_count, 1))
            colors.append(color + "99")

    if len(ids) <= 1:
        return ""

    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="remainder",
        marker=dict(colors=colors),
        textinfo="label",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value} concepts<extra></extra>",
    ))
    fig.update_layout(
        title="Gioia Data Structure — Hierarchy",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Georgia,serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return _to_html(fig)


def concept_frequency_chart(concepts: List[FirstOrderConcept], top_n: int = 25) -> str:
    counter: Counter = Counter(c.concept for c in concepts)
    top = counter.most_common(top_n)
    if not top:
        return ""
    labels = [t[0] for t in top]
    freqs = [t[1] for t in top]
    fig = go.Figure(go.Bar(
        x=freqs, y=labels, orientation="h",
        marker_color="#2E86AB", text=freqs, textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} First-Order Concept Frequencies",
        xaxis_title="Frequency",
        height=max(400, top_n * 22 + 100),
        margin=dict(l=20, r=50, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia,serif", size=11),
        yaxis=dict(autorange="reversed"),
    )
    return _to_html(fig)


def dimension_wordcloud(dim: AggregateDimension) -> str:
    tokens: List[str] = []
    for theme in dim.second_order_themes:
        tokens.extend([theme.name.replace(" ", "_")] * 3)
        for foc in theme.first_order_concepts:
            tokens.extend(foc.verbatim.split())
    text = " ".join(tokens) if tokens else dim.name
    try:
        wc = WordCloud(
            width=900, height=280, background_color="white",
            colormap="viridis", max_words=50, collocations=False,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(dim.name, fontsize=13, fontweight="bold", pad=8)
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def generate_all_gioia(result: GioiaResult) -> Dict:
    viz: Dict = {}
    if result.aggregate_dimensions:
        viz["data_structure"] = data_structure_table(result.aggregate_dimensions)
        viz["sunburst"] = hierarchy_sunburst(result.aggregate_dimensions)
        viz["wordclouds"] = {}
        for dim in result.aggregate_dimensions:
            wc = dimension_wordcloud(dim)
            if wc:
                viz["wordclouds"][dim.id] = wc
    if result.first_order_concepts:
        cf = concept_frequency_chart(result.first_order_concepts)
        if cf:
            viz["concept_frequency"] = cf
    return viz
