#!/usr/bin/env python3
"""Generate the reader-facing Colab training/export companion notebook."""

from pathlib import Path

import nbformat as nbf


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "cane_v1_training_export.ipynb", "provenance": []},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# YOLO26n Cane V1 — Training and Export\n\n"
            "## Goal\n\n"
            "Train a deterministic YOLO26n smoke run or Cane V1 dataset, validate "
            "the best checkpoint, export ONNX and NCNN artifacts, and compare "
            "validation metrics. Colab values prove workflow correctness and model "
            "quality only; they are not Raspberry Pi performance measurements."
        ),
        nbf.v4.new_markdown_cell(
            "## Setup\n\n"
            "For the checked-in smoke validation, keep `DATASET = 'coco8.yaml'` and "
            "`EPOCHS = 1`. For V1, upload the harmonized dataset archive, extract it "
            "under `/content/cane-v1`, set `DATASET` to its `data.yaml`, and choose "
            "the reviewed training duration."
        ),
        nbf.v4.new_code_cell(
            "# @title 1. Install the pinned major-version runtime\n"
            "%pip install -q 'ultralytics==8.4.132'"
        ),
        nbf.v4.new_code_cell(
            "# @title 2. Parameters\n"
            "MODEL = 'yolo26n.pt'\n"
            "DATASET = 'coco8.yaml'\n"
            "EPOCHS = 1\n"
            "IMAGE_SIZE = 320\n"
            "BATCH = 8\n"
            "SEED = 42\n"
            "RUN_NAME = 'notebook-gpu-smoke'"
        ),
        nbf.v4.new_code_cell(
            "# @title 3. Verify hardware and runtime\n"
            "import json, platform, shutil\n"
            "from pathlib import Path\n"
            "import torch, ultralytics\n"
            "from ultralytics import YOLO\n\n"
            "assert torch.cuda.is_available(), 'This notebook validation requires a GPU runtime'\n"
            "DEVICE = 0\n"
            "print({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, "
            "'ultralytics': ultralytics.__version__})"
        ),
        nbf.v4.new_markdown_cell("## Steps"),
        nbf.v4.new_code_cell(
            "# @title 4. Train deterministically\n"
            "PROJECT = Path('/content/yolo-pi-notebook-runs')\n"
            "model = YOLO(MODEL)\n"
            "train_result = model.train(data=DATASET, epochs=EPOCHS, imgsz=IMAGE_SIZE, "
            "batch=BATCH, device=DEVICE, workers=2, project=str(PROJECT), name=RUN_NAME, "
            "exist_ok=True, seed=SEED, deterministic=True, plots=True, verbose=False)\n"
            "BEST = Path(train_result.save_dir) / 'weights' / 'best.pt'\n"
            "assert BEST.exists(), BEST\n"
            "BEST"
        ),
        nbf.v4.new_code_cell(
            "# @title 5. Validate and export\n"
            "best_model = YOLO(str(BEST))\n"
            "pt_metrics = best_model.val(data=DATASET, imgsz=IMAGE_SIZE, device=DEVICE, "
            "split='val', plots=False, verbose=False)\n"
            "ONNX = Path(YOLO(str(BEST)).export(format='onnx', imgsz=IMAGE_SIZE, "
            "batch=1, device='cpu'))\n"
            "NCNN = Path(YOLO(str(BEST)).export(format='ncnn', imgsz=IMAGE_SIZE, batch=1, "
            "device='cpu', end2end=False, quantize=16))\n"
            "assert ONNX.exists() and NCNN.exists()\n"
            "ONNX, NCNN"
        ),
        nbf.v4.new_markdown_cell("## Checks"),
        nbf.v4.new_code_cell(
            "# @title 6. Check exported-model metric parity\n"
            "def metrics_dict(metrics):\n"
            "    return {str(k): float(v) for k, v in metrics.results_dict.items() "
            "if isinstance(v, (int, float))}\n\n"
            "onnx_metrics = YOLO(str(ONNX)).val(data=DATASET, imgsz=IMAGE_SIZE, "
            "device='cpu', split='val', plots=False, verbose=False)\n"
            "ncnn_metrics = YOLO(str(NCNN)).val(data=DATASET, imgsz=IMAGE_SIZE, "
            "device='cpu', split='val', plots=False, verbose=False)\n"
            "SUMMARY = {'purpose': 'workflow validation; not Pi performance', "
            "'gpu': torch.cuda.get_device_name(0), 'model': MODEL, 'dataset': DATASET, "
            "'epochs': EPOCHS, 'imgsz': IMAGE_SIZE, 'seed': SEED, "
            "'metrics': {'pytorch': metrics_dict(pt_metrics), "
            "'onnx': metrics_dict(onnx_metrics), 'ncnn': metrics_dict(ncnn_metrics)}}\n"
            "SUMMARY_PATH = PROJECT / RUN_NAME / 'validation-summary.json'\n"
            "SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2, sort_keys=True) + '\\n')\n"
            "SUMMARY"
        ),
        nbf.v4.new_code_cell(
            "# @title 7. Package artifacts\n"
            "ARCHIVE = Path(shutil.make_archive('/content/yolo-pi-notebook-artifacts', "
            "'zip', PROJECT))\n"
            "print({'archive': str(ARCHIVE), 'summary': str(SUMMARY_PATH)})"
        ),
        nbf.v4.new_markdown_cell(
            "## Next Steps\n\n"
            "1. Download the archive before stopping the Colab session.\n"
            "2. For V1, review the harmonizer report and manually inspect every class.\n"
            "3. Record model-quality metrics separately from Pi runtime metrics.\n"
            "4. Validate all exported artifacts again on the Raspberry Pi using the fixed benchmark manifest."
        ),
    ]
    output = Path("notebooks/cane_v1_training_export.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
