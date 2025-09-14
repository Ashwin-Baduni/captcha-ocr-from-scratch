# Neural CAPTCHA Recognition System

## Abstract

This repository implements a neural approach to CAPTCHA text extraction, addressing the limitations of classical OCR methods when faced with variable fonts, noisy backgrounds, and conditional text rendering. The system comprises synthetic dataset generation, CNN-based classification, and sequence-to-sequence models with attention mechanisms for robust text extraction from CAPTCHA images.

## 1. Introduction

Traditional OCR systems rely on handcrafted features that fail under the diverse conditions present in modern CAPTCHAs. This implementation presents a neural architecture pipeline that:
- Generates synthetic CAPTCHA datasets with increasing complexity
- Implements classification models for fixed vocabulary recognition
- Develops sequence-to-sequence models for variable-length text extraction
- Handles conditional rendering based on visual cues

## 2. Architecture Overview

### Task 0: Dataset Generation
- **Easy Set**: Fixed font (DejaVu Sans), white background, consistent capitalization
- **Hard Set**: 6 font families, noisy backgrounds (Gaussian, salt-pepper, speckle), variable capitalization
- **Bonus Set**: Conditional rendering - green background (normal), red background (reversed display)

### Task 1: Classification
- **LightweightCNN**: 3 convolutional blocks, suitable for easy dataset
- **ImprovedCNN**: 5 convolutional blocks with residual connections, handles complex patterns

### Task 2: Text Extraction (OCR)
- **Architecture**: CNN encoder (ResNet-based) + LSTM decoder with attention
- **Loss Function**: Cross-entropy for seq2seq, CTC loss for alignment-free training
- **Decoding**: Beam search with configurable beam width

### Task 3: Conditional Rendering
- Specialized training on bonus dataset
- Learns invariance to display transformations
- Maintains consistent label extraction regardless of visual rendering

## 3. Installation

### Prerequisites
```bash
Python 3.8+
CUDA 11.0+ (optional, for GPU acceleration)
```

### Setup
```bash
git clone https://github.com/username/captcha-ocr.git
cd captcha-ocr
pip install -r requirements.txt
```

## 4. Usage

### Quick Start
```bash
# Run complete pipeline (all tasks)
python code/pipeline_corrected.py --num-images 1000

# Run with visualization
python code/pipeline_corrected.py --num-images 1000 --visualize

# Clean previous runs
python code/pipeline_corrected.py --clean --num-images 1000
```

### Individual Tasks
```bash
# Task 0: Generate datasets
python -m code.task0.generate --num-images 1000

# Task 1: Train classifier
python -m code.task1.train --dataset easy --model lightweight --epochs 30

# Task 2: Train OCR
python -m code.task2.train_generation --dataset easy --model seq2seq --epochs 50

# Task 3: Train on bonus set
python -m code.task3.bonus_generation --model seq2seq --epochs 30
```

### Pipeline Options
```bash
--num-images N      # Images per dataset (default: 100)
--tasks 0 1 2 3    # Specific tasks to run
--timeout S         # Timeout per task in seconds
--no-visualize      # Disable visualization generation
--clean             # Clean previous outputs

# Epoch Configuration (NEW)
--epochs N                    # Set all epochs to N (overrides individual settings)
--task1-epochs E H B          # Task 1 epochs: Easy Hard Bonus (default: 10 50 30)
--task2-epochs E H            # Task 2 epochs: Easy Hard (default: 20 30)
--task3-epochs N              # Task 3 epochs (default: 25)
```

### Epoch Configuration Examples
```bash
# Quick test with 5 epochs for all tasks
python code/pipeline_corrected.py --epochs 5

# Custom epochs per task
python code/pipeline_corrected.py --task1-epochs 15 30 20 --task2-epochs 25 35 --task3-epochs 30

# Fast training on easy dataset only
python code/pipeline_corrected.py --task1-epochs 5 0 0 --task2-epochs 10 0 --tasks 1 2

# Production training with more epochs
python code/pipeline_corrected.py --task1-epochs 100 150 100 --task2-epochs 80 100 --task3-epochs 75
```

## 5. Project Structure

```
captcha_ocr/
├── code/
│   ├── task0/
│   │   ├── base_generator.py      # Abstract generator class
│   │   ├── easy_generator.py      # Easy dataset generation
│   │   ├── hard_generator.py      # Hard dataset with noise
│   │   ├── bonus_generator.py     # Conditional rendering
│   │   └── effects.py             # Image augmentation effects
│   ├── task1/
│   │   ├── model.py               # CNN architectures
│   │   ├── dataset.py             # Classification dataset loader
│   │   └── train.py               # Training pipeline
│   ├── task2/
│   │   ├── generation.py          # Seq2seq models
│   │   ├── dataset_seq2seq.py     # OCR dataset handling
│   │   └── train_generation.py    # OCR training
│   ├── task3/
│   │   └── bonus_generation.py    # Conditional rendering handler
│   ├── utils/
│   │   ├── config.py              # Configuration parameters
│   │   └── comprehensive_visualizer.py  # Visualization tools
│   └── pipeline_corrected.py      # Main orchestration script
├── data/                           # Generated datasets
├── results/
│   ├── models/                    # Trained model checkpoints
│   ├── reports/                   # Training logs and metrics
│   └── visualizations/            # Generated visualizations
└── requirements.txt
```

## 6. Visualizations

The pipeline generates comprehensive visualizations:
- `task0_generation_overview.png`: Dataset samples and word distribution
- `task1_training_curves.png`: Loss and accuracy progression
- `task2_ocr_results.png`: CER/WER metrics comparison
- `task2_ocr_predictions.png`: Sample predictions vs ground truth
- `task3_bonus_results.png`: Condition-specific performance
- `task3_condition_comparison.png`: Green vs Red rendering analysis
- `final_summary.png`: Complete pipeline results overview

## 7. Key Features

- **Modular Design**: Independent task modules with clear interfaces
- **Automatic Mixed Precision**: FP16 training support for efficiency
- **Data Augmentation**: Online augmentation during training
- **Checkpoint Management**: Automatic best model selection
- **Comprehensive Logging**: TensorBoard integration, JSON reports
- **Error Analysis**: Detailed failure case documentation

## 8. Hyperparameters

### Training Configuration
```python
Learning Rate: 1e-3 (with ReduceLROnPlateau)
Batch Size: 32
Optimizer: Adam (β1=0.9, β2=0.999)
Weight Decay: 1e-4
Dropout: 0.3 (Easy), 0.5 (Hard/Bonus)
```

### Data Generation
```python
Image Size: 200x60 pixels
Font Sizes: 20-36pt
Noise Levels: 0-2% (Gaussian σ)
Rotation: ±5 degrees
Shear: ±0.2
```

## 9. Reproducibility

All experiments use fixed random seeds:
```python
Random Seed: 42
PyTorch: torch.manual_seed(42)
NumPy: np.random.seed(42)
```
