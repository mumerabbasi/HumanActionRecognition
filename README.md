# Multi-View Furniture Assembly Action Segmentation

This repository hosts a **multi-view dataset** collected specifically for **human action segmentation** tasks in the domain of **furniture assembly**. The dataset is captured from **8 synchronized Intel RealSense D435i cameras**, providing multiple viewpoints for each recorded action sequence.

## Project: Projektpraktikum Human Activity Understanding

This dataset and associated code were developed as part of the **Projektpraktikum Human Activity Understanding**. The core of this work is a **spatio-temporal model** which recognizes actions across both space and time, leveraging multi-view data for a more robust understanding of complex human-object interactions.

## Data Composition

- **5 different actions** related to furniture assembly.  
- Each action sequence has **8 distinct camera views**.  
- Annotations are provided for **View 5** (using the VGG Annotator tool), which align directly with frames in other views.

## Spatio-Temporal Action Recognition Model

Our approach centers on a **spatio-temporal transformer**:

- **Multihead Self-Attention** aggregates features across **both space and time** from all 8 camera views.  
- This **focuses on critical interactions** from multiple viewpoints, addressing occlusions or subtle details.  
- By incorporating **temporal information**, the model captures the evolution of actions over time, leading to stronger performance on assembly tasks.

### Why Multi-View + Spatio-Temporal Attention?

- **Holistic Perspective**: Multiple camera angles provide comprehensive coverage, reducing blind spots.  
- **Temporal Dynamics**: The spatio-temporal backbone learns **how actions progress** frame by frame.  
- **Rich Feature Fusion**: Multihead attention integrates important cues from each viewpoint at each timestep.

## Results

- Achieved **88.4% frame-level classification accuracy** (i.e., an action prediction for each frame).  
- Measured an approximate **0.82 macro-F1 score** across all action classes.  
- These metrics are on a frame-by-frame comparison basis, referred to as **frame-level accuracy**.

## Download & Setup

1. **Download** the dataset from [Google Drive](https://drive.google.com/file/d/1ArpoIoj6K5msEck54Qc2nmbfTys39UEO/view?usp=drive_link).  
2. **Unzip** the dataset to a directory of your choice.  
3. **Use** the `preprocess_data.py` script to organize the data into train, validation, and test splits.

## Training the Spatio-Temporal Transformer

1. **Install dependencies** (Python 3.8+, PyTorch, torchvision, etc.).  
2. **Configure** hyperparameters (e.g., number of transformer layers, attention heads, sequence lengths) in `config.yaml`.  
3. **Train** the multi-view spatio-temporal model by running the `train.py` script. Logs and checkpoints will be stored in the specified `output_dir`.

## Applications

- **Spatio-Temporal Action Recognition**  
  - Leverage multiple camera viewpoints and temporal data for enhanced accuracy.
- **3D Pose Estimation**  
  - Synchronized captures are useful for 3D reconstructions.
- **Computer Vision Research**  
  - Explore advanced attention mechanisms for robust classification.

## License

This work is released under the [MIT License](LICENSE).

## Contributing

We welcome contributions! If you have ideas for improvement, feel free to open a pull request or issue.

---

**Happy exploring and learning from this multi-view, spatio-temporal furniture assembly dataset!**
