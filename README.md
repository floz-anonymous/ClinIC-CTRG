#ClinIC-CTRG: Contrastive Alignment with Knowledge-augmentation and In-Context Learning for CT Report Generation

This repository contains the official implementation of **ClinIC-CTRG**, a two-stage framework for generating structured radiology reports. This work leverages chest CT-scans, contrastive learning, knowledge augmentation & In-Context Learning to improve diagnostic accuracy.

## Project Structure

```
├── Stage 1/
│ ├── inter-model-contrastive-alignment.py
│ ├── intra-model-contrastive-learning.py
│ ├── knowledgege-augmented-report-generation.py
│ ├── strong-weak-multimodal-fusion.py
│ ├── text_encoder.py
│ └── visual_encoder.py
├── Stage 2/
│ ├── evaluation.py
│ └── ICL.py
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- NVIDIA GPU (Recommended for training)

## 4. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Stage 1**

```bash
python3 "Stage 1/knowledge-augmented-report-generation.py" --data_dir /path/to/images \
  --annotations /path/to/ann.csv --batch_size <batch_size> --epochs <epochs> --learning_rate <learning_rate> \
  --checkpoint_dir ./checkpoints/stage1
```

**Stage 2**

```bash
python3 "Stage 2/ICL.py" \
  --vlm_client <vlm_client> \
  --checkpoint_path ./checkpoints/stage2 \
```

**Evaulation**

```bash
python3 "Stage 2/evaluation.py" \
  --predicted_reports <predicted_reports> \
  --actual_reports <actual_reports>
```

## License

This project is released under the **MIT License**.
