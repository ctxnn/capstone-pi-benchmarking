#!/usr/bin/env python3
"""Fail if an executed notebook is structurally invalid or contains error cells."""

import argparse
from pathlib import Path

import nbformat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    unexecuted = [index for index, cell in enumerate(code_cells) if cell.execution_count is None]
    errors = [
        (index, output.get("ename"), output.get("evalue"))
        for index, cell in enumerate(code_cells)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    # The Colab CLI currently writes all execution_count fields as null and also
    # suppresses output from setup-only cells. Require outputs from every
    # evidence-producing cell when that recognizable representation is present.
    colab_cli_style = len(unexecuted) == len(code_cells)
    evidence_cells_missing_output = []
    if colab_cli_style:
        evidence_cells_missing_output = [
            index
            for index, cell in enumerate(code_cells)
            if index >= 2 and not cell.get("outputs")
        ]
    if errors or (unexecuted and not colab_cli_style) or evidence_cells_missing_output:
        raise RuntimeError(
            "unexecuted_code_cells="
            f"{unexecuted}, error_outputs={errors}, "
            f"evidence_cells_missing_output={evidence_cells_missing_output}"
        )
    print(
        {
            "valid": True,
            "code_cells": len(code_cells),
            "error_outputs": 0,
            "execution_count_mode": "colab-cli-null" if colab_cli_style else "jupyter",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
