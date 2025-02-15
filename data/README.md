# 📌 Dataset Description

This dataset was collected using **8 synchronized Intel RealSense D435i cameras**, capturing multiple sequences from **8 different viewpoints**. Below are the key details:

## 📷 Camera Setup
- 8 Intel RealSense D435i cameras
- Synchronized for consistent multi-view capture

## 📂 Data Composition
- Each sequence includes **8 distinct views** (one per camera)
- A total of **5 different actions** related to **furniture assembly** are recorded

## 🏷️ Annotations
- Provided for **View 5**, created using the **VGG Annotator tool**
- Since all cameras are synchronized, these annotations can be directly applied to corresponding frames in other views
- Ensures **consistent labeling** across all viewpoints

## 📥 Download Dataset
- Download the dataset from [here](https://drive.google.com/file/d/1ArpoIoj6K5msEck54Qc2nmbfTys39UEO/view?usp=drive_link)
- Unzip the data in this directory
- Use `preprocess_data.py` to divide it into **train, validation, and test** subsets

## 🎯 Applications
This dataset is ideal for:
- Multi-view action recognition
- 3D pose estimation
- Computer vision applications
- **Furniture assembly analysis**

🚀 **Start exploring and building with this dataset!**
