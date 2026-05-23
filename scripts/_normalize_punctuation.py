"""One-off: replace em dashes throughout the project with ASCII punctuation,
and rename the project title from 'safety-sae-feature-forecasting' to
'Early-Layer-SAE-Feature-Forecasting' in title positions.

Run from repo root:

    python scripts/_normalize_punctuation.py

Em-dash rules (in order of precedence):
  1. " - " at start of a heading line ending in "— DONE" / "— PENDING" / etc.:
     replace with " - " (hyphen with surrounding spaces).
  2. " — " inside backtick label patterns ("`x.py` — description"): replace
     with ": " (colon-space) — these are label:description constructs.
  3. " — " elsewhere in prose: replace with "; " (semicolon-space) — these
     are typically clause joiners or elaborations of independent clauses.
  4. "—" without surrounding spaces (rare): replace with "-".

Project-name rename only touches title positions:
  - README.md H1
  - docs/implementation_plan.md H1
  - notebook H1s (cell 0 of each .ipynb)
  - src/__init__.py module docstring
  - the embedded H1 markdown inside scripts/_build_safety_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

OLD = "safety-sae-feature-forecasting"
NEW = "Early-Layer-SAE-Feature-Forecasting"

EM = "—"  # em dash


def replace_em_dashes(text: str) -> tuple[str, int]:
    """Return (replaced_text, count_replaced)."""
    out_lines = []
    n = 0
    for line in text.split("\n"):
        stripped = line.lstrip()
        # Rule 1: heading line "## ... — STATUS"
        if stripped.startswith("#") and f" {EM} " in line:
            new_line = line.replace(f" {EM} ", " - ")
            n += line.count(f" {EM} ")
            out_lines.append(new_line)
            continue
        # Rule 2: backtick-label patterns "`x` — y" → "`x` - y" (label/desc)
        # We use "- " (hyphen-space) because this is a label-description
        # construct, not a clause joiner.
        pattern_label = re.compile(r"(`[^`]+`)\s+—\s+")
        before = line
        line = pattern_label.sub(r"\1 - ", line)
        n += (before.count(f" {EM} ") - line.count(f" {EM} "))
        # Rule 3: remaining " — " in prose → "; "
        if f" {EM} " in line:
            n += line.count(f" {EM} ")
            line = line.replace(f" {EM} ", "; ")
        # Rule 4: bare em dash without spaces (rare)
        if EM in line:
            n += line.count(EM)
            line = line.replace(EM, "-")
        out_lines.append(line)
    return "\n".join(out_lines), n


def process_text_file(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    orig = text
    text, n_em = replace_em_dashes(text)
    n_name = 0
    # Project-name rename: only in specific anchored contexts so we don't
    # accidentally touch path identifiers like 'safety-sae-cache'.
    # Targets:
    #   line 1: '# safety-sae-feature-forecasting'  (README H1)
    #   line 1: '# Implementation plan: safety-sae-feature-forecasting'
    #   docstring opener: '"""safety-sae-feature-forecasting'
    targets = [
        (f"# {OLD}\n", f"# {NEW}\n"),
        (f"# {OLD}", f"# {NEW}"),
        (f"# Implementation plan: {OLD}", f"# Implementation plan: {NEW}"),
        (f'"""{OLD}', f'"""{NEW}'),
    ]
    for src, dst in targets:
        if src in text:
            n_name += text.count(src)
            text = text.replace(src, dst)
    if text != orig:
        path.write_text(text, encoding="utf-8")
    return n_em, n_name


def process_notebook(path: Path) -> tuple[int, int]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    n_em_total, n_name_total = 0, 0
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        for i, line in enumerate(src):
            new, n_em = replace_em_dashes(line)
            # In notebook H1 cells, rename the project title prefix.
            for old_t, new_t in [
                (f"# {OLD}: Step", f"# {NEW}: Step"),
                (f"# {OLD}", f"# {NEW}"),
            ]:
                if old_t in new:
                    n_name_total += new.count(old_t)
                    new = new.replace(old_t, new_t)
            if new != line:
                src[i] = new
            n_em_total += n_em
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return n_em_total, n_name_total


def main():
    targets_text = [
        REPO / "README.md",
        REPO / "docs" / "implementation_plan.md",
        REPO / "docs" / "03_activation_cache.md",
        REPO / "docs" / "04_probes.md",
        REPO / "docs" / "05_data_efficiency.md",
        REPO / "docs" / "06_generalization.md",
        REPO / "docs" / "07_writeup.md",
        REPO / "src" / "__init__.py",
        REPO / "src" / "probe.py",
        REPO / "src" / "data_efficiency.py",
        REPO / "scripts" / "step4_train_probes.py",
        REPO / "scripts" / "step5_analysis.py",
        REPO / "scripts" / "step7_make_figures.py",
        REPO / "scripts" / "check_safety_cache.py",
        REPO / "scripts" / "_build_safety_notebook.py",
    ]
    targets_nb = [
        REPO / "notebooks" / "01_smoke_test.ipynb",
        REPO / "notebooks" / "02_activation_cache.ipynb",
        REPO / "notebooks" / "03_safety_cache.ipynb",
    ]
    total_em, total_name = 0, 0
    for p in targets_text:
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        em, name = process_text_file(p)
        if em or name:
            print(f"  {p.relative_to(REPO)}: {em} em-dashes, {name} title rename(s)")
        total_em += em
        total_name += name
    for p in targets_nb:
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        em, name = process_notebook(p)
        if em or name:
            print(f"  {p.relative_to(REPO)}: {em} em-dashes, {name} title rename(s)")
        total_em += em
        total_name += name
    print(f"Done. {total_em} em-dashes replaced, {total_name} title renames.")


if __name__ == "__main__":
    main()
