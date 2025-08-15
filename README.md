# Visual Guidance System

A comprehensive deep learning system for detecting and selecting spatial transition points (STPs) in game screenshots. This system combines object detection with neural selection to identify the most important navigation points for visual guidance.

## Overview

The Visual Guidance System is designed to assist in game navigation by:
1. **Detecting STPs**: Identifying all spatial transition points in game screenshots using Faster R-CNN
2. **Selecting MSTPs**: Choosing the most important STP using a dual-branch neural network
3. **Retrieval Augmentation**: Enhancing predictions through similarity-based feature retrieval

## Architecture

### Core Components

- **STP Detector**: Faster R-CNN with FPN backbone for detecting candidate regions
- **MSTP Selector**: Dual-branch network combining local and global context features
- **Adapter Modules**: Efficient fine-tuning mechanisms for domain adaptation
- **Feature Bank**: Retrieval augmentation system for improved performance

### Model Architecture

```
Input Image → STP Detector → Candidate Regions → MSTP Selector → Selected MSTP
                    ↓              ↓                    ↓
              Faster R-CNN    Bounding Boxes    Dual-Branch Network
              + FPN + Adapter   + Confidence     + Global Context
```

## Repository Structure

```
VisualGuidance/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/
│   └── config.py            # Configuration parameters
├── models/
│   ├── __init__.py          # Package initialization
│   ├── adapters.py          # Adapter module implementations
│   ├── stp_detector.py      # STP detection model
│   └── mstp_selector.py     # MSTP selection model
├── pipeline/
│   ├── __init__.py          # Package initialization
│   └── inference.py         # Complete inference pipeline
├── utils/
│   ├── __init__.py          # Package initialization
│   └── feature_bank.py      # Feature bank for retrieval
├── scripts/
│   ├── train_stp_detector.py    # STP detector training
│   ├── train_mstp_selector.py   # MSTP selector training
│   └── demo.py                  # Demonstration script
├── data/
│   ├── annotations/         # Original annotation files
│   ├── processed/           # Processed dataset files
│   └── datasets/            # Image datasets
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone [your-repo-url]
   cd VisualGuidance
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## Usage

### Single Image Inference

```python
from pipeline.inference import VisualGuidancePipeline

# Initialize pipeline
pipeline = VisualGuidancePipeline(
    detector_model_path="path/to/stp_detector.pth",
    selector_model_path="path/to/mstp_selector.pth",
    use_retrieval=True,
    retrieval_bank_file="path/to/feature_bank.pkl"
)

# Process single image
processed_img, detected_boxes, selected_index = pipeline.process_single_image(
    "path/to/image.jpg", score_threshold=0.7
)

print(f"Detected {len(detected_boxes)} STPs")
print(f"Selected MSTP index: {selected_index}")
```

### Batch Processing

```python
# Process directory of images
results = pipeline.process_directory(
    input_dir="path/to/images/",
    output_dir="path/to/results/",
    score_threshold=0.7
)
```

## Training

### Training STP Detector

```bash
python scripts/train_stp_detector.py \
    --annotations_file data/annotations/stp_labels.json \
    --images_dir data/datasets/images/ \
    --output_dir outputs/stp_detector/ \
    --num_epochs 100 \
    --batch_size 2 \
    --use_adapter \
    --adapter_dim 256
```

### Training MSTP Selector

```bash
python scripts/train_mstp_selector.py \
    --annotations_file data/processed/model2_train_dataset.json \
    --images_dir data/datasets/images/ \
    --output_dir outputs/mstp_selector/ \
    --num_epochs 100 \
    --batch_size 8 \
    --use_adapter \
    --bottleneck_dim 256
```

### Building Feature Bank

```bash
python utils/feature_bank.py \
    --annotations data/annotations/stp_labels.json \
    --images_dir data/datasets/images/ \
    --output feature_bank.pkl \
    --top_k 1000
```

## Configuration

The system configuration is centralized in `config/config.py`:

```python
from config.config import Config

# STP Detector settings
detector_config = Config.STP_DETECTOR_CONFIG
print(f"Detection threshold: {detector_config['score_threshold']}")

# MSTP Selector settings
selector_config = Config.MSTP_SELECTOR_CONFIG
print(f"Crop size: {selector_config['crop_size']}")
```

### Key Parameters

- **Detection Threshold**: Confidence threshold for STP detection (default: 0.7)
- **Adapter Dimensions**: Bottleneck dimensions for fine-tuning (default: 256)
- **Retrieval Alpha**: Weight for retrieval vs model scores (default: 0.7)
- **Batch Sizes**: Training batch sizes for each model
- **Learning Rates**: Optimizer learning rates

## Datasets

### Supported Games

- **Dark Souls Series**: DS1, DS2, DS3
- **Elden Ring**
- **BMW**

### Data Format

#### Annotations (stp_labels.json)
```json
[
    {
        "image_id": "20250208191922_1.jpg",
        "MSTP": [393, 422, 524, 481],
        "STP": [[1306, 394, 1412, 500]]
    }
]
```

#### Processed Datasets
- **model1_train_dataset.json**: Training data for STP detector
- **model2_train_dataset.json**: Training data for MSTP selector
- **bmw_*.json**: BMW-specific datasets

## Applications

### Game Navigation
- Identify key path points in complex game environments
- Provide visual guidance for optimal routing
- Support real-time navigation assistance

### Research Applications
- Computer vision research in gaming contexts
- Domain adaptation studies
- Multi-task learning with adapters

## Experiments

### Demo Script

```bash
python scripts/demo.py \
    --image path/to/test_image.jpg \
    --detector_model path/to/stp_detector.pth \
    --selector_model path/to/mstp_selector.pth \
    --use_retrieval \
    --feature_bank feature_bank.pkl
```

### Evaluation

The system provides comprehensive evaluation metrics:
- **STP Detector**: mAP, IoU, Precision, Recall
- **MSTP Selector**: Accuracy, Top-k accuracy
- **Pipeline**: End-to-end performance metrics

## Technical Details

### Adapter Fine-tuning

The system implements efficient fine-tuning through adapter modules:
- **Bottleneck Architecture**: Low-rank adaptation for parameter efficiency
- **Residual Connections**: Preserve pre-trained knowledge
- **Domain Adaptation**: Quick adaptation to new game environments

### Retrieval Augmentation

Feature-based retrieval enhances MSTP selection:
- **Cosine Similarity**: Measure feature similarity
- **Quality Scoring**: Rank features by informativeness
- **Fusion Strategy**: Combine retrieval and model scores


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
