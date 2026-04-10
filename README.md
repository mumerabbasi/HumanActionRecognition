<div align="center">

# Multi-View Furniture Assembly Human Action Recognition 
Spatio-temporal action recognition for furniture assembly using synchronized multi-camera RGB video, ResNet-50 spatial features, view attention, and a temporal Transformer classifier.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![ResNet-50](https://img.shields.io/badge/ResNet--50-Spatial_Features-4B5563)
![Multi-View](https://img.shields.io/badge/Multi--View-8_Cameras-0F766E)
![Transformer](https://img.shields.io/badge/Transformer-Temporal_Model-2563EB)
![Dataset](https://img.shields.io/badge/Dataset-Furniture_Assembly-16A34A)

</div>

---

## Highlights

| Capability | Detail |
|:-----------|:-------|
| Camera setup | **8 synchronized Intel RealSense D435i cameras** |
| Action scope | **5 furniture-assembly actions** plus a background/no-action label |
| Spatial encoder | **ResNet-50** ImageNet features with a 2048-dim pooled output |
| View fusion | **Multi-head attention** across camera views |
| Temporal reasoning | **Transformer encoder** over frame sequences |
| Reported results | **88.4% frame-level accuracy** and about **0.82 macro-F1** |

---

## Demo

https://github.com/user-attachments/assets/1338eb24-245e-4dfc-b77d-1aadb9dfff61

The local demo asset is kept in [`assets/HAU_demo.mp4`](assets/HAU_demo.mp4).

---

## What It Does

This project recognizes furniture-assembly actions from synchronized multi-view video. Each sample contains aligned frame sequences from eight RealSense D435i cameras, allowing the model to reduce occlusion sensitivity and capture human-object interactions from multiple angles.

The model predicts a frame-sequence action label by:

1. Extracting per-frame spatial features from every view with ResNet-50
2. Applying multi-head attention to weight and fuse evidence across views
3. Modeling how the action evolves over time with a Transformer encoder
4. Pooling the sequence representation and classifying the action

---

## Architecture

```
        ┌─────────────────────────────────────┐
        │  8 Synchronized Camera Views        │
        │  Intel RealSense D435i Streams      │
        └──────────────────┬──────────────────┘
                           │ RGB frame sequences
                           ▼
        ┌─────────────────────────────────────┐
        │  ResNet-50 Spatial Encoder          │
        │  2048-dim pooled frame features     │
        └──────────────────┬──────────────────┘
                           │ Per-view embeddings
                           ▼
        ┌─────────────────────────────────────┐
        │  Multi-Head View Attention          │
        │  Cross-view feature fusion          │
        └──────────────────┬──────────────────┘
                           │ View-aware sequence features
                           ▼
        ┌─────────────────────────────────────┐
        │  Temporal Transformer Encoder       │
        │  Action dynamics over time          │
        └──────────────────┬──────────────────┘
                           │ Temporal representations
                           ▼
        ┌─────────────────────────────────────┐
        │  Mean Pooling Across Time & Views   │
        │  Sequence-level representation      │
        └──────────────────┬──────────────────┘
                           │ Classification head input
                           ▼
        ┌─────────────────────────────────────┐
        │  Action Classifier                  │
        │  Furniture assembly prediction      │
        └─────────────────────────────────────┘
```

The public model APIs are:

- `SpatialFeatureExtractor(pretrained=True)`
- `MultiviewActionRecognitionModel(...)`

---

## Results

| Metric | Reported Value |
|:-------|:--------------:|
| Frame-level accuracy | **88.4%** |
| Macro-F1 | **~0.82** |

The metrics are computed frame by frame against the action annotations. The evaluation script reports accuracy, macro-F1, loss, and a confusion matrix for the selected split.

---

## Dataset

The dataset was collected for the Projektpraktikum Human Activity Understanding project using **8 synchronized Intel RealSense D435i cameras**. Each sequence contains eight aligned views, one per camera.

Download the dataset from Google Drive:

https://drive.google.com/file/d/1ArpoIoj6K5msEck54Qc2nmbfTys39UEO/view?usp=drive_link

Annotation details:

- Labels were created on **View 5** with the VGG Annotator tool.
- Because all cameras are synchronized, View 5 annotations align with corresponding frames from the other seven views.
- [`data/annotations/action_labels.csv`](data/annotations/action_labels.csv) stores temporal segments, frame ranges, action labels, sequence IDs, view IDs, and FPS-derived frame differences.
- [`data/annotations/classes.json`](data/annotations/classes.json) includes `no action` plus five assembly actions: inserting vertical short rods, inserting elbows, inserting horizontal short rods, inserting horizontal long rods, and flip.

---

## Getting Started

```bash
git clone https://github.com/mumerabbasi/HumanActionRecognition.git
cd HumanActionRecognition

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the downloaded dataset under `data/raw/`, then preprocess it into train, validation, and test splits:

```bash
python -m data.preprocess_data \
    --source_dir data/raw \
    --dest_dir data/processed \
    --train_pct 0.70 \
    --val_pct 0.15 \
    --test_pct 0.15
```

---

## Training

Configuration lives in [`configs/default.yaml`](configs/default.yaml). The default setup trains the ResNet-50 plus view-attention plus temporal-Transformer model on `data/processed`.

```bash
python -m src.training.train --config configs/default.yaml
```

Training logs and checkpoints are written under the configured `output_dir`.

---

## Evaluation

Evaluate a saved checkpoint on the configured split:

```bash
python -m src.training.evaluate \
    --config configs/default.yaml \
    --checkpoint output/custom_models/run_YYYYMMDD_HHMMSS/best_model_epoch_10.pth
```

Evaluation logs are written under `output/evaluation_results` by default.

---

## Project Structure

```
HumanActionRecognition/
|-- assets/
|   `-- HAU_demo.mp4
|-- configs/
|   `-- default.yaml
|-- data/
|   |-- annotations/
|   |   |-- action_labels.csv
|   |   `-- classes.json
|   `-- preprocess_data.py
|-- src/
|   |-- data/
|   |   |-- dataset.py
|   |   `-- transforms.py
|   |-- models/
|   |   |-- attention_views.py
|   |   |-- multiview_action_recognition_model.py
|   |   |-- spatial_feature_extractor.py
|   |   `-- transformer_encoder_temporal.py
|   |-- training/
|   |   |-- evaluate.py
|   |   `-- train.py
|   `-- utils/
|       |-- helper.py
|       |-- logger.py
|       `-- preprocess_data_utils.py
`-- requirements.txt
```

---

## Applications

| Area | Use |
|:-----|:----|
| Assembly assistance | Recognize progress and mistakes during furniture assembly |
| Human activity understanding | Study fine-grained human-object interaction from synchronized views |
| Robotics | Provide action context for collaborative or assistive robots |
| Computer vision research | Benchmark multi-view attention and temporal sequence modeling |

---

## License

This project is released under the [MIT License](LICENSE).
