"""Run dependency-free structural and evidence checks for this repository."""

from __future__ import annotations

import ast
import json
import re
import sys
import tokenize
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "env", ".conda", "__pycache__"}

CORE_NOTEBOOK = ROOT / "CoreML" / "Google Colab" / "SAiDL_(2).ipynb"
SSM_NOTEBOOK = ROOT / "SSM" / "Google Colab" / "SSM.ipynb"
CORE_REPORT = ROOT / "Reports" / "TRACK1_APL.md"
SSM_REPORT = ROOT / "Reports" / "TRACK2_SSM.md"

CORE_RESULTS = {
    "CE": ("85.56%", "79.90%", "73.18%", "39.92%"),
    "NCE": ("83.46%", "75.76%", "61.03%", "34.66%"),
    "FL": ("84.26%", "79.05%", "71.49%", "41.89%"),
    "NFL": ("79.98%", "71.32%", "56.98%", "35.08%"),
    "NCE+MAE": ("90.14%", "85.15%", "78.05%", "49.55%"),
    "NCE+RCE": ("83.83%", "77.87%", "67.59%", "37.01%"),
    "NFL+MAE": ("89.83%", "84.22%", "75.62%", "49.58%"),
    "NFL+RCE": ("84.23%", "77.92%", "68.47%", "37.71%"),
}


def _walk_files(suffix: str):
    for path in ROOT.rglob(f"*{suffix}"):
        if not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            yield path


def _output_text(notebook: dict[str, Any]) -> str:
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                chunks.append("".join(text) if isinstance(text, list) else str(text))
            data = output.get("data", {})
            plain = data.get("text/plain") if isinstance(data, dict) else None
            if plain:
                chunks.append("".join(plain) if isinstance(plain, list) else str(plain))
            trace = output.get("traceback")
            if trace:
                chunks.append("\n".join(trace))
    return "\n".join(chunks)


def _check_python(errors: list[str]) -> int:
    count = 0
    for path in _walk_files(".py"):
        count += 1
        try:
            with tokenize.open(path) as handle:
                ast.parse(handle.read(), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"Python parse failed for {path.relative_to(ROOT)}: {exc}")
    return count


def _load_notebooks(errors: list[str]) -> dict[Path, dict[str, Any]]:
    notebooks: dict[Path, dict[str, Any]] = {}
    for path in _walk_files(".ipynb"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Notebook JSON failed for {path.relative_to(ROOT)}: {exc}")
            continue
        if payload.get("nbformat") not in {4} or not isinstance(payload.get("cells"), list):
            errors.append(f"Unexpected notebook structure: {path.relative_to(ROOT)}")
        notebooks[path] = payload
    return notebooks


def _check_core_evidence(notebook: dict[str, Any], errors: list[str]) -> None:
    output = _output_text(notebook)
    report = CORE_REPORT.read_text(encoding="utf-8")
    for loss_name, values in CORE_RESULTS.items():
        values_pattern = r"\s+".join(map(re.escape, values))
        notebook_pattern = rf"(?m)^{re.escape(loss_name)}\s+{values_pattern}\s*$"
        if not re.search(notebook_pattern, output):
            errors.append(f"CoreML notebook output is missing the recorded {loss_name} result row")
        report_row = f"| {loss_name} | {' | '.join(values)} |"
        if report_row not in report:
            errors.append(f"Track 1 report is missing or changed for {loss_name}")


def _check_ssm_evidence(notebook: dict[str, Any], errors: list[str]) -> None:
    output = _output_text(notebook)
    report = SSM_REPORT.read_text(encoding="utf-8")
    expected = ("Best Test Accuracy: 86.83%", "Epoch 200/200 | Loss: 0.9280 | Acc: 86.63%")
    for value in expected:
        if value not in output:
            errors.append(f"SSM notebook output is missing: {value}")
    for value in ("86.83%", "86.63%", "0.9280"):
        if value not in report:
            errors.append(f"Track 2 report is missing recorded value: {value}")


def _check_generated_artifacts(errors: list[str]) -> None:
    forbidden_dirs = [ROOT / ".vscode"]
    forbidden_files = list(ROOT.rglob("*.pyc"))
    for path in forbidden_dirs:
        if path.exists():
            errors.append(f"Generated/editor directory should not be committed: {path.relative_to(ROOT)}")
    for path in forbidden_files:
        if not any(part in {".venv", "venv", "env", ".conda"} for part in path.parts):
            errors.append(f"Bytecode artifact should not be committed: {path.relative_to(ROOT)}")


def main() -> int:
    errors: list[str] = []
    python_count = _check_python(errors)
    notebooks = _load_notebooks(errors)

    if CORE_NOTEBOOK not in notebooks:
        errors.append(f"Missing primary notebook: {CORE_NOTEBOOK.relative_to(ROOT)}")
    else:
        _check_core_evidence(notebooks[CORE_NOTEBOOK], errors)

    if SSM_NOTEBOOK not in notebooks:
        errors.append(f"Missing primary notebook: {SSM_NOTEBOOK.relative_to(ROOT)}")
    else:
        _check_ssm_evidence(notebooks[SSM_NOTEBOOK], errors)

    _check_generated_artifacts(errors)

    if errors:
        print("Repository checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"OK: parsed {python_count} Python files and {len(notebooks)} notebooks")
    print("OK: report values match the stored CoreML and SSM notebook outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
