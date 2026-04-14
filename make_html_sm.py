"""Convert supplementary-material.ipynb to HTML with wide-table fix."""
import re
import subprocess

NOTEBOOK = "supplementary-material.ipynb"
OUTPUT = "supplementary-material.html"

subprocess.run(
    ["jupyter-nbconvert", "--to", "html", "--no-input", NOTEBOOK],
    check=True,
)

with open(OUTPUT, encoding="utf-8") as f:
    html = f.read()

# Directly patch table-layout: fixed -> auto in the nbconvert CSS block.
# This avoids CSS specificity fights entirely.
html = re.sub(
    r"(\.jp-RenderedHTMLCommon\s+table\s*\{[^}]*)table-layout:\s*fixed",
    r"\1table-layout: auto",
    html,
    flags=re.DOTALL,
)

# Patch jp-OutputArea-child: display:table doesn't support overflow scrolling.
# Change to display:block + overflow-x:auto so wide tables get a scrollbar.
html = re.sub(
    r"(\.jp-OutputArea-child\s*\{[^}]*)display:\s*table",
    r"\1display: block",
    html,
    flags=re.DOTALL,
)
html = re.sub(
    r"(\.jp-OutputArea-child\s*\{[^}]*)overflow:\s*hidden",
    r"\1overflow-x: auto",
    html,
    flags=re.DOTALL,
)

# Disable MathJax automatic line-breaking — it splits equations across lines.
html = re.sub(
    r"(linebreaks\s*:\s*\{[^}]*automatic\s*:\s*)true",
    r"\1false",
    html,
)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Written: {OUTPUT}")
