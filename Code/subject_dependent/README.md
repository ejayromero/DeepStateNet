# Neural Network Training Scripts

This repository contains modular scripts for training neural networks on EEG microstate data with K-fold cross-validation.

## 📁 File Structure

```
Code/
├── subject_dependent/
│   ├── ms.py           # MicroSNet models training
│   ├── dcn.py          # DeepConvNet training
│   └── lib/
│       ├── my_functions.py         # Data loading utilities
│       ├── my_models.py            # Model architectures
│       └── my_models_functions.py  # Shared training/plotting functions
Data/
└── [EEG data files]
Output/
├── ica_rest_all/
│   ├── modkmeans_results/
│   │   ├── ms_timeseries/          # One-hot encoded microstate data
│   │   └── modk_sequence/          # Categorical microstate data
│   └── dependent/                  # Results saved here
└── [other output folders]
```

## 🚀 Quick Start

### 1. Training MicroSNet Models

```bash
# Basic usage with default microsnet
python ms.py

# Train attention model with custom parameters
python ms.py --model-name attention_microsnet --epochs 150 --lr 0.001

# Train with different number of subjects and folds
python ms.py --model-name multiscale_microsnet --n-subjects 30 --n-folds 5
```

### 2. Training DeepConvNet

```bash
# Basic DCN training
python dcn.py

# Custom parameters
python dcn.py --epochs 200 --batch-size 64 --lr 0.0005
```

## 📊 Available Models

### MicroSNet Variants (`ms.py`)

| Model Name | Input Format | Description |
|------------|-------------|-------------|
| `microsnet` | One-hot | Simple temporal CNN for microstate sequences |
| `multiscale_microsnet` | One-hot | Multi-scale temporal CNN with parallel branches |
| `embedded_microsnet` | Categorical | Embedding-based CNN for categorical sequences |
| `attention_microsnet` | Categorical | Transformer-enhanced hierarchical CNN with attention |

### DeepConvNet (`dcn.py`)

- **Input**: Raw EEG signals (channels × timepoints)
- **Architecture**: Deep convolutional network from braindecode
- **Use case**: Standard EEG classification baseline

## ⚙️ Command Line Arguments

### Common Arguments (Both Scripts)

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 32 | Training batch size |
| `--epochs` | 100 (ms.py), 100 (dcn.py) | Number of training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--n-subjects` | 50 | Number of subjects to process |
| `--n-folds` | 4 | Number of CV folds |
| `--type-of-subject` | dependent | Subject type: dependent/independent/adaptive |
| `--seed` | 42 | Random seed for reproducibility |
| `--no-cuda` | False | Disable CUDA (use CPU) |

### MicroSNet Specific (`ms.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | microsnet | Model architecture to use |
| `--n-clusters` | 12 | Number of microstate clusters |
| `--ms-file-specific` | indiv | Microstate file variant |
| `--dropout` | 0.25 | Dropout rate |
| `--embedding-dim` | 64 | Embedding dimension (attention models) |
| `--transformer-layers` | 4 | Number of transformer layers |
| `--transformer-heads` | 8 | Number of attention heads |

## 🔧 Advanced Usage Examples

### Hyperparameter Tuning

```bash
# Low learning rate with more epochs
python ms.py --model-name attention_microsnet --lr 1e-4 --epochs 200

# Higher dropout for regularization
python ms.py --model-name microsnet --dropout 0.4 --batch-size 16

# Larger transformer architecture
python ms.py --model-name attention_microsnet --transformer-layers 6 --transformer-heads 12 --embedding-dim 128
```

### Different Data Configurations

```bash
# Different number of microstate clusters
python ms.py --n-clusters 8 --model-name microsnet

# Different subject analysis type
python ms.py --type-of-subject independent --model-name attention_microsnet

# More robust cross-validation
python ms.py --n-folds 10 --model-name multiscale_microsnet
```

### Resource Management

```bash
# CPU-only training
python ms.py --no-cuda --batch-size 16

# Smaller batch size for memory constraints
python ms.py --batch-size 8 --model-name attention_microsnet

# Process fewer subjects for testing
python ms.py --n-subjects 5 --epochs 50
```

## 📈 Output and Results

### Saved Files

**Results**: `Output/ica_rest_all/{type_of_subject}/{type_of_subject}_{model_name}_cv_{n_folds}fold_results/`

- `{type_of_subject}_{model_name}_cv_{n_folds}fold_results.npy` - Complete results data
- `{type_of_subject}_{model_name}_CV_test_metrics.png` - CV vs test performance
- `{type_of_subject}_{model_name}_CV_training_curves.png` - Training curves
- `{type_of_subject}_{model_name}_avg_confusion_matrix.png` - Confusion matrix

### Result Metrics

- **Balanced Accuracy**: Accounts for class imbalance
- **F1 Macro**: Average F1 score across all classes
- **Cross-Validation**: K-fold CV with stratified splits
- **Test Performance**: Final evaluation on held-out 10% of data

## 🔄 Data Flow

### Automatic Data Loading

1. **MicroSNet**: Automatically selects data format based on model:
   - `microsnet`, `multiscale_microsnet` → `ms_timeseries` (one-hot)
   - `embedded_microsnet`, `attention_microsnet` → `modk_sequence` (categorical)

2. **DeepConvNet**: Loads raw EEG data directly

### Training Pipeline

1. **Data Split**: 90% for CV, 10% for final test
2. **Cross-Validation**: K-fold stratified CV on the 90%
3. **Model Selection**: Best fold based on validation balanced accuracy
4. **Final Test**: Evaluate best model on held-out 10%

## 🛠️ Troubleshooting

### Common Issues

1. **Low Accuracy (~30% - worse than 37.5% random baseline!)**:
   ```bash
   # This indicates a serious problem - debug with:
   python ms.py --n-subjects 1 --epochs 50  # Test single subject
   
   # Check class distribution
   print(f"Class distribution: {torch.unique(y, return_counts=True)}")
   print(f"Random baseline: 37.5% (due to 50%/25%/25% class imbalance)")
   ```

2. **Memory Issues**:
   ```bash
   # Reduce batch size and use CPU
   python ms.py --batch-size 8 --no-cuda
   ```

3. **File Not Found**:
   - Check `Output/ica_rest_all/modkmeans_results/` exists
   - Verify microstate data files are present
   - Ensure correct `--ms-file-specific` parameter

4. **CUDA Issues**:
   ```bash
   # Force CPU usage
   python ms.py --no-cuda
   ```

### Resume Training

Both scripts automatically resume from existing results:
```bash
# Will continue from where it left off
python ms.py --model-name microsnet
```

### Debug Mode

For debugging specific subjects:
```bash
# Train only first subject with verbose output
python ms.py --n-subjects 1 --epochs 10 --log-interval 1
```

## 📋 Requirements

- Python 3.11.9
- PyTorch
- scikit-learn
- numpy, pandas
- matplotlib, seaborn
- braindecode 0.8.1 (for DCN)
- Pycrostates (for ModKmeans)

## 🎯 Expected Performance

### Class Distribution
- **Rest (Class 0)**: 50% of data
- **Open (Class 1)**: 25% of data  
- **Close (Class 2)**: 25% of data

### Performance Baselines
- **Random Baseline**: 37.5% (due to class imbalance: $0.5² + 0.25² + 0.25² = 37.5%$)
- **Majority Class Baseline**: 50% (always predict Rest)

### Typical Results
- **DeepConvNet**: 60-80% balanced accuracy
- **MicroSNet**: 40-70% balanced accuracy (varies by model)
- **Warning**: Performance below 37.5% indicates the model is worse than random!

### Class Labels
- **0: Rest** - Right hand resting state
- **1: Open** - Right hand open state  
- **2: Close** - Right hand close state

## 📞 Support

If you encounter issues:

1. Check file paths and data availability
2. Try with reduced parameters (`--n-subjects 1 --epochs 10`)
3. Use CPU mode (`--no-cuda`) to rule out GPU issues
4. Verify model architecture compatibility with your data