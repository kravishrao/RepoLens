"""Generate the complete 10-page IEEE format Mini Project Report for RepoLens.

Run:
    python docs/generate_report_pdf.py

Produces:
    docs/RepoLens_Mini_Project_Report.pdf (Exactly 10 pages)
"""

import subprocess
from pathlib import Path
from pypdf import PdfReader

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parent
TEMPLATE_PATH = REPORT_DIR / "RepoLens_Mini_Project_Report_template.html"
HTML_PATH = REPORT_DIR / "RepoLens_Mini_Project_Report.html"
PDF_PATH = REPORT_DIR / "RepoLens_Mini_Project_Report.pdf"

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def get_browser_path() -> str:
    if Path(EDGE_PATH).exists():
        return EDGE_PATH
    if Path(CHROME_PATH).exists():
        return CHROME_PATH
    raise FileNotFoundError("Neither Chrome nor Edge was found.")

def generate_report():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Calibrated parameters for exact 10-page IEEE layout
    params = {
        "font_size_pt": 9.55,
        "line_height": 1.28,
        "col_gap_in": 0.22,
        "p_margin_rem": 0.34,
        "heading_margin_rem": 0.74,
    }

    rendered = template
    rendered = rendered.replace("__FONT_SIZE_PT__", str(params["font_size_pt"]))
    rendered = rendered.replace("__LINE_HEIGHT__", str(params["line_height"]))
    rendered = rendered.replace("__COL_GAP_IN__", str(params["col_gap_in"]))
    rendered = rendered.replace("__P_MARGIN_REM__", str(params["p_margin_rem"]))
    rendered = rendered.replace("__HEADING_MARGIN_REM__", str(params["heading_margin_rem"]))

    HTML_PATH.write_text(rendered, encoding="utf-8")

    browser = get_browser_path()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH.resolve()}",
        HTML_PATH.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)

    reader = PdfReader(str(PDF_PATH))
    total_pages = len(reader.pages)
    print(f"Generated {PDF_PATH} ({total_pages} pages, {PDF_PATH.stat().st_size:,} bytes)")
    assert total_pages == 10, f"Expected 10 pages, got {total_pages}"
    return total_pages

if __name__ == "__main__":
    generate_report()
