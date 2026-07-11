"""CSV / PDF biodiversity report generation from an aggregated pipeline result."""
import pathlib

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_csv(result: dict, out_path: pathlib.Path) -> None:
    # Explicit columns so the header row still appears when a list is empty --
    # pd.DataFrame([]) otherwise has no columns to infer, producing a blank
    # section with no header instead of a clearly-empty table.
    species_df = pd.DataFrame(result["species_summary"], columns=["species", "unique_count", "avg_confidence", "rarity"])
    review_df = pd.DataFrame(result["review_queue"], columns=["track_id", "species_guess", "max_confidence", "timestamp_sec", "crop_path"])

    with open(out_path, "w", newline="") as f:
        f.write("Species summary\n")
        species_df.to_csv(f, index=False)
        f.write("\nFlagged for manual review (low confidence)\n")
        review_df.to_csv(f, index=False)


def generate_pdf(result: dict, out_path: pathlib.Path, video_name: str, job_id: str) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    story = [
        Paragraph("DiveBuddy Biodiversity Report", styles["Title"]),
        Paragraph(f"Source video: {video_name}", styles["Normal"]),
        Paragraph(f"Job ID: {job_id}", styles["Normal"]),
        Spacer(1, 16),
        Paragraph("Species observed", styles["Heading2"]),
    ]

    species_rows = [["Species", "Unique count", "Avg. confidence", "Rarity"]]
    for row in result["species_summary"]:
        species_rows.append([
            row["species"], str(row["unique_count"]), f"{row['avg_confidence']:.2f}", row["rarity"],
        ])
    if len(species_rows) == 1:
        species_rows.append(["No species identified above the confidence threshold", "", "", ""])

    species_table = Table(species_rows, hAlign="LEFT")
    species_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d2240")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ]))
    story.append(species_table)

    story.append(Spacer(1, 20))
    story.append(Paragraph("Flagged for manual review (low confidence)", styles["Heading2"]))
    review_rows = [["Timestamp (s)", "Best guess", "Max confidence"]]
    for row in result["review_queue"]:
        review_rows.append([str(row["timestamp_sec"]), row["species_guess"], f"{row['max_confidence']:.2f}"])
    if len(review_rows) == 1:
        review_rows.append(["Nothing flagged", "", ""])

    review_table = Table(review_rows, hAlign="LEFT")
    review_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d2240")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ]))
    story.append(review_table)

    doc.build(story)
