# Deepfake Detection Using EfficientNet-B4 and Optical Flow Analysis

## Project Overview

This project focuses on detecting manipulated (deepfake) videos using deep learning and computer vision techniques. It uses the **Celeb-DF (Celeb-DeepFake) dataset** to distinguish between authentic and manipulated video content.

The project implements an EfficientNet-B4-based deepfake detection model for frame-level binary classification and incorporates facial landmark analysis, facial-region extraction, and optical flow-based motion analysis to investigate spatial and temporal characteristics of real and fake videos.

The implementation is developed in Python using PyTorch, OpenCV, MediaPipe, and other computer vision and data processing libraries.

## Objectives

- Detect real and manipulated facial content using a deep learning-based classification model.
- Extract video frames for image-based model training.
- Identify facial landmarks and extract important facial regions.
- Analyze facial motion between consecutive frames using dense optical flow.
- Generate optical flow heatmaps to visualize motion patterns.
- Extract motion statistics and compare real and fake video characteristics.

## Dataset

This project uses the **Celeb-DF dataset**, a publicly documented deepfake video dataset developed for deepfake detection research.

The dataset contains authentic celebrity videos and manipulated videos generated using deepfake techniques.

### Dataset Access

The original Celeb-DF dataset is not included in this repository due to its size and access restrictions.

To obtain the dataset:

1. Visit the official Celeb-DF repository.
2. Submit the dataset access request form.
3. Wait for approval from the dataset authors.
4. Download the dataset using the authorized download link.

**Official Dataset Resources:**

- [Celeb-DF Official Repository](https://github.com/yuezunli/celeb-deepfakeforensics)
- [Dataset Access Request Form](https://forms.gle/2jYBby6y1FBU3u6q9)

### Dataset Organization

After downloading and extracting the dataset, organize the video files into the following directory structure. The names below reflect the paths used in the notebook.

```text
Celeb-DF/
│
├── Real/
│   └── Celeb-real/
│       ├── video_001.mp4
│       ├── video_002.mp4
│       └── ...
│
├── Fake/
│   └── New folder/
│       ├── video_001.mp4
│       ├── video_002.mp4
│       └── ...
│
├── Real Frames/
├── Fake Frames/
│
├── Real Frame Face Segmentation/
├── Fake Face Segmentation/
│
├── Real Frame Landmark/
│
├── heatmaps_real/
├── heatmaps_fake/
│
├── frame_labels.csv
├── landmark_summary.csv
├── real_motion.csv
├── fake_motion.csv
└── combined_motion.csv
```

The generated directories and CSV files are created during the notebook's processing stages.

### Dataset Processing Summary

In the recorded notebook execution, the initial video inspection reported:

| Category | Number of videos |
|---|---:|
| Real videos | 408 |
| Fake videos | 408 |
| Total videos | 816 |

The frame-label generation stage produced the following recorded counts:

| Category | Extracted frames |
|---|---:|
| Real frames | 175,289 |
| Fake frames | 172,315 |
| Total frames | 347,604 |

These figures describe the recorded project processing run and are not a claim about the complete official Celeb-DF dataset.

## Methodology

The project consists of the following processing and analysis stages.

### 1. Video Frame Extraction

The original real and fake video files are processed using OpenCV.

The implementation:
- Reads videos from the respective real and fake directories.
- Extracts frames from video sequences.
- Uses a target frame rate setting of 35 FPS for frame sampling.
- Resizes extracted frames to 256 × 256 pixels.
- Saves frames as JPEG images.
- Organizes extracted frames into individual video folders.

The extracted frames are subsequently used for image preprocessing and classification.

### 2. Facial Landmark Detection

MediaPipe Face Mesh is used to identify facial landmarks in the extracted frames.

The implementation uses selected landmark indices corresponding to important facial regions, including:

- Eyes
- Nose
- Lips

Detected landmarks are visualized on facial images using OpenCV.

The annotated images are saved for further inspection and facial feature analysis.

### 3. Facial Region Extraction

The project implements facial-region extraction using MediaPipe Face Mesh and OpenCV.

Nine facial regions are defined:

1. Left eye
2. Right eye
3. Left eyebrow
4. Right eyebrow
5. Left cheek
6. Right cheek
7. Nose
8. Mouth
9. Jaw

The implementation uses selected facial landmark coordinates to calculate bounding boxes and crop the corresponding regions.

The extracted facial-region images are saved separately for real and fake frames.

This processing provides localized facial images for examining potential manipulation-related visual characteristics.

### 4. Frame Labelling and Dataset Preparation

The extracted frames are organized into a labelled image dataset.

Binary labels are assigned as follows:

| Label | Category |
|---|---|
| 0 | Real |
| 1 | Fake |

A CSV file named `frame_labels.csv` is generated with two columns:

- `filepath`: Location of the extracted image.
- `label`: Corresponding real or fake classification label.

The generated CSV contains 347,604 recorded frame entries.

A custom PyTorch Dataset class loads the images and their corresponding labels during training.

### 5. Image Preprocessing

Before being passed into the EfficientNet-B4 model, images undergo the following transformations:

- Resizing to 380 × 380 pixels.
- Conversion into PyTorch tensors.
- Normalization using ImageNet mean and standard deviation.

The normalization parameters are:

```python
Mean = [0.485, 0.456, 0.406]

Standard Deviation = [0.229, 0.224, 0.225]
```

The prepared dataset is divided into training and validation subsets using an 80:20 random split.

### 6. EfficientNet-B4 Deepfake Detection Model

The primary classification model uses the EfficientNet-B4 architecture with pretrained ImageNet weights.

The model is implemented using the `efficientnet-pytorch` library.

#### Model Architecture

The implemented architecture consists of:

1. Pretrained EfficientNet-B4 backbone for feature extraction.
2. Global Average Pooling for feature dimensionality reduction.
3. Fully connected layer with 512 neurons.
4. ReLU activation.
5. Dropout with a probability of 0.4.
6. Final fully connected layer with one output neuron.

The EfficientNet-B4 feature extractor produces 1792-dimensional pooled features before classification.

The final output represents a binary classification logit, which is converted into a probability using the sigmoid function during prediction.

#### Training Configuration

| Parameter | Configuration |
|---|---|
| Architecture | EfficientNet-B4 |
| Pretrained weights | ImageNet |
| Input image size | 380 × 380 |
| Batch size | 32 |
| Number of epochs | 5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Focal Loss |
| Focal Loss alpha | 0.25 |
| Focal Loss gamma | 2.0 |
| Learning rate scheduler | Cosine Annealing |
| Scheduler T_max | 5 |
| Dropout | 0.4 |
| Training precision | Automatic Mixed Precision |
| Backbone | Frozen during training |

The training implementation uses PyTorch automatic mixed precision to support GPU-based training.

The Focal Loss function is implemented using binary cross-entropy with logits and a focusing parameter to emphasize more difficult classification examples.

### 7. Model Training and Validation

The model is trained using the prepared training dataset and evaluated on the validation dataset after each epoch.

The training process records:

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy

The model parameters are saved in the following file:

```text
efficientnet_texture_artifact.pth
```

The saved weights can subsequently be loaded into the EfficientNet-B4 model architecture for inference.

### 8. Optical Flow-Based Motion Analysis

In addition to image-based classification, the project implements optical flow analysis to investigate motion patterns between consecutive video frames.

The implementation uses the Farneback dense optical flow algorithm provided by OpenCV.

The processing pipeline includes:

1. Loading consecutive extracted frames.
2. Converting frames into grayscale images.
3. Computing dense optical flow between consecutive frames.
4. Calculating motion magnitude and direction.
5. Extracting motion statistics.
6. Generating color-coded optical flow heatmaps.

The optical flow calculation uses the following Farneback parameters:

| Parameter | Value |
|---|---:|
| Pyramid scale | 0.5 |
| Pyramid levels | 3 |
| Window size | 15 |
| Iterations | 3 |
| Polynomial neighbourhood | 5 |
| Polynomial sigma | 1.2 |

The current implementation uses a full-face mask for the optical flow feature calculation.

### 9. Optical Flow Heatmap Generation

Optical flow heatmaps are generated to visualize the magnitude and direction of motion between consecutive frames.

The flow vectors are converted into HSV representations and subsequently saved as color-coded images.

The heatmaps are organized into separate directories:

```text
heatmaps_real/
heatmaps_fake/
```

These visualizations provide an additional means of examining motion differences between real and manipulated video sequences.

### 10. Motion Feature Extraction

The project extracts statistical information from optical flow heatmaps.

The implemented motion statistics include:

- Mean motion intensity.
- Standard deviation of motion intensity.

The extracted statistics are saved in CSV format.

The following output files are generated:

| File | Description |
|---|---|
| `real_motion.csv` | Motion statistics for real-video heatmaps |
| `fake_motion.csv` | Motion statistics for fake-video heatmaps |
| `combined_motion.csv` | Combined real and fake motion statistics |

### 11. Motion Visualization and Comparison

Matplotlib and Seaborn are used to visualize and compare motion statistics.

The notebook implements:

- Real-versus-fake motion line plots.
- Box plots for motion distribution.
- Kernel Density Estimation (KDE) plots.
- Motion comparison plots for extracted feature columns.

These visualizations help examine differences in motion intensity between authentic and manipulated video frames.

## Experimental Results

The recorded EfficientNet-B4 training and validation results are presented below.

### Training Performance

| Epoch | Training Loss | Training Accuracy |
|---|---:|---:|
| 1 | 0.0113 | 93.44% |
| 2 | 0.0076 | 95.64% |
| 3 | 0.0059 | 96.66% |
| 4 | 0.0046 | 97.43% |
| 5 | 0.0039 | 97.87% |

### Validation Performance

| Epoch | Validation Loss | Validation Accuracy |
|---|---:|---:|
| 1 | 0.0061 | 96.29% |
| 2 | 0.0036 | 97.89% |
| 3 | 0.0023 | 98.71% |
| 4 | 0.0017 | 99.07% |
| 5 | 0.0013 | 99.25% |

**Final recorded validation accuracy: 99.25%.**

These values are taken from the notebook's recorded training execution. They represent validation performance, not independent test-set performance or guaranteed generalization to unseen datasets.

## Technologies and Tools Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| CNN Architecture | EfficientNet-B4 |
| Computer Vision | OpenCV |
| Facial Landmark Detection | MediaPipe Face Mesh |
| Image Processing | PIL, NumPy |
| Data Processing | Pandas |
| Numerical Computation | NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Progress Monitoring | tqdm |
| Model Training | CUDA, Automatic Mixed Precision |
| Development Environment | Jupyter Notebook / Visual Studio Code |

## Hardware Configuration

The recorded model training environment used:

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- PyTorch Version: 1.9.1+cu111
- CUDA Version: 11.1

GPU availability and compatibility depend on the installed PyTorch and CUDA versions.

## Repository Structure

```text
Deepfake-Detection/
│
├── Deep Detection.ipynb
│
├── README.md
│
├── requirements
