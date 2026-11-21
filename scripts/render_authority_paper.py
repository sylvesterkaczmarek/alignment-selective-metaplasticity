"""Render the research report from its Markdown source using ReportLab (optional)."""

import argparse
import gzip
from html import escape
import json
from pathlib import Path
from urllib.parse import urljoin
import re

from reportlab.graphics.shapes import Drawing, Line, PolyLine, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/blob/v0.3.1/"


def html(paragraph):
    def link(m):
        url = m[2]
        if not url.startswith("https://"):
            url = urljoin(BASE + "docs/", url)
        return f'<link href="{url}" color="#125965">{m[1]}</link>'

    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, escape(paragraph))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)


def trajectory():
    with gzip.open(ROOT / "results/authority-synthetic-v1/runs.jsonl.gz", "rt") as f:
        row = next(
            r
            for line in f
            if (r := json.loads(line))["condition"] == "overlapping"
            and r["method"] == "ewc"
            and r["trial"]["model_seed"] == 811
        )
    chart = Drawing(450, 104)
    x0, y0, w, h = 32, 19, 378, 65
    low = 0.70

    def y(v):
        return y0 + (v - low) / (1 - low) * h

    for v in (0.7, 0.8, 0.9, 1.0):
        chart.add(
            Line(
                x0,
                y(v),
                x0 + w,
                y(v),
                strokeColor=colors.HexColor("#d4dce0"),
                strokeWidth=0.5,
            )
        )
        chart.add(String(4, y(v) - 3, f"{v:.1f}", fontName="Helvetica", fontSize=8))
    for i, colour in enumerate(("#16736e", "#a44918")):
        points = []
        for step, values in enumerate(row["trajectory"]):
            points.extend([x0 + w * step / (len(row["trajectory"]) - 1), y(values[i])])
        chart.add(
            PolyLine(points, strokeColor=colors.HexColor(colour), strokeWidth=1.8)
        )
    for i in (0, 3, 6, 9, 12):
        chart.add(
            String(x0 + w * i / 12 - 2, 5, str(i), fontName="Helvetica", fontSize=8)
        )
    chart.add(
        String(
            x0,
            92,
            "Authorised acceptance",
            fontSize=8,
            fillColor=colors.HexColor("#16736e"),
        )
    )
    chart.add(
        String(
            152,
            92,
            "Unauthorised resistance",
            fontSize=8,
            fillColor=colors.HexColor("#a44918"),
        )
    )
    return chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    doc = SimpleDocTemplate(
        str(args.out),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="Measuring correction authority across model updates",
        author="Sylvester Kaczmarek",
        invariant=1,
    )

    def style(name, font, size, leading, space):
        return ParagraphStyle(
            name,
            fontName=font,
            fontSize=size,
            leading=leading,
            spaceAfter=space,
            textColor=colors.HexColor("#1c2c35"),
        )

    title = style("Title", "Times-Bold", 22, 25, 8)
    head = style("Section", "Helvetica-Bold", 11, 15, 4)
    body = style("Body", "Times-Roman", 10.1, 13.2, 7)
    small = style("Note", "Helvetica", 8, 11, 8)
    content = []
    text = (ROOT / "docs/authority-study.md").read_text().split("## Reproduction")[0]
    for para in text.strip().split("\n\n"):
        if para.startswith("# "):
            content.append(Paragraph(escape(para[2:]), title))
        elif para.startswith("## "):
            if para == "## Open-weight outcomes":
                content.append(PageBreak())
            if para == "## Resources and scope":
                content.append(trajectory())
                content.append(
                    Paragraph(
                        "Figure 1. Paired correction accuracy at each capability epoch in overlapping-feature EWC block 811.",
                        small,
                    )
                )
            content.append(Paragraph(escape(para[3:]), head))
        elif para.startswith("|"):
            rows = [
                [x.strip() for x in line.strip("|").split("|")]
                for line in para.splitlines()
                if "---" not in line
            ]
            rows[0] = [
                "Method",
                "SmolLM2\nacquisition",
                "SmolLM2\nearlier retention",
                "Qwen\nacquisition",
                "Qwen\nearlier retention",
            ]
            table = Table(rows, colWidths=[108, 76, 88, 72, 88], hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("LEADING", (0, 0), (-1, -1), 10),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        (
                            "LINEBELOW",
                            (0, 0),
                            (-1, 0),
                            0.75,
                            colors.HexColor("#236561"),
                        ),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
            content.extend([table, Spacer(1, 8)])
        else:
            content.append(
                Paragraph(html(para), small if para.startswith("**Sylvester") else body)
            )
    content.append(
        Paragraph(
            html(
                "[Protocol, data and reproduction commands]("
                + BASE
                + "docs/authority-study.md)"
            ),
            small,
        )
    )

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#51616b"))
        canvas.drawString(
            20 * mm, 10 * mm, "Alignment-Selective Metaplasticity   /   0.3.1"
        )
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, str(document.page))

    doc.build(content, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
