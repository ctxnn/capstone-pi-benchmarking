# Cane V1 presentation showcase

These are qualitative outputs from the fixed held-out test fixture, not training images. Inference used the accepted `best.pt` checkpoint at 640 px, FP32, confidence 0.25, IoU 0.7.

## Full held-out quality result (1,515 images)

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Pretrained YOLO26n | 0.2933 | 0.2332 | 0.2413 | 0.1749 |
| Fine-tuned YOLO26n-Cane V1 | **0.5991** | **0.5100** | **0.5272** | **0.3632** |

## Suggested narration

- Cane V1 expands beyond generic COCO objects toward mobility hazards such as poles, stairs, signboards, ground obstacles, and two-wheelers.
- The examples show behavior in crowded, mixed-traffic, and obstacle-focused scenes.
- These images illustrate capability; the defensible quantitative claim comes from the complete 1,515-image held-out evaluation.
- Raspberry Pi latency, FPS, RAM, CPU, temperature, and camera timing remain device-side measurements and are not inferred from laptop results.
