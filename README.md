# Salt Dome Detection in Seismic Data Using a True 3D Global Attention Convolutional Network



Official implementation of **"Salt Dome Detection in Seismic Data Using a True 3D Global Attention Convolutional Network with Self-Supervised Denoising Pretext Training"** by Matin Mahzad and Majid Bagheri.

## 🔬 Overview

This repository presents the application of **unfactorized 3D global attention** for seismic salt dome segmentation on the Netherlands F3 block (Zechstein salt formation). The architecture combines volumetric convolution for local feature extraction with true global self-attention for complete spatial context modeling — enabling accurate delineation of complex salt body geometries from diapir stem to overhanging canopy.

### Key Innovation

Unlike windowed attention mechanisms (e.g., Swin Transformer) or factorized attention that fragment volumetric structures, our approach maintains **complete 3D global attention** throughout the network (every-voxel-to-every-voxel). This is especially critical for salt dome segmentation, where boundaries are volumetrically extended structures spanning hundreds to thousands of voxels.

## 🎯 Key Features

- **True 3D Global Attention**: Unfactorized attention mechanism (every-voxel-to-every-voxel)
- **Self-Supervised Pretext Training**: Multi-survey denoising across Kerry-3D, Opunake-3D, and Kahu-3D
- **Discriminative Transfer Learning**: Layer-wise learning rate decay (LLRD) with gradual unfreezing
- **Advanced Loss Function**: Unified Focal Loss for extreme class imbalance (~1.48% foreground ratio)
- **Slanted Triangular LR Scheduling**: Optimized learning rate trajectory for transfer learning
- **Mixed Precision Training**: Efficient GPU utilization with automatic mixed precision + gradient checkpointing

## 📊 Performance

### Netherlands F3 Block — Zechstein Salt (Alaudah et al., 2019)

Metrics are computed at the voxel level with decision threshold τ = 0.5. MCC is the primary imbalance-robust summary metric given the extreme foreground scarcity (~1.48% of voxels).

| Metric | Description |
|--------|-------------|
| **Dice** | Region overlap measure |
| **IoU** | Intersection over union (Jaccard Index) |
| **Precision** | Positive predictive value |
| **Recall** | True positive rate |
| **MCC** | Matthews Correlation Coefficient (primary metric) |
| **F2** | Recall-weighted F-score |
| **Balanced Accuracy** | Average of sensitivity and specificity |
| **Cohen's Kappa** | Agreement beyond chance |

## 🏗️ Architecture

```
Hybrid U-Net with 3D Global Attention
├── Encoder
│   ├── Encoder Block 1: 3D Conv + Global Attention (128³)
│   ├── Downsampling (128³ → 64³)
│   ├── Encoder Block 2: 3D Conv + Global Attention (64³)
│   └── Downsampling (64³ → 32³)
├── Bridge
│   └── Bridge Block: 3D Conv + Global Attention (32³)
├── Decoder
│   ├── Upsampling (32³ → 64³)
│   ├── Decoder Block 2: 3D Conv + Global Attention (64³) + Skip Connection
│   ├── Upsampling (64³ → 128³)
│   └── Decoder Block 1: 3D Conv + Global Attention (128³) + Skip Connection
└── Output: 1×1×1 Convolution
```

### Global Attention Mechanism

```python
Q = Conv3D(x)  # Query projection
K = Conv3D(x)  # Key projection
V = Conv3D(x)  # Value projection

Attention = Softmax(Q @ K^T / √d_k)  # Global attention map
Output = Attention @ V                 # Attended features
```

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8 (for GPU training)
NumPy, SciPy
```

### Installation

```bash
# Clone repository
git clone https://github.com/Matin-Mahzad/salt-dome-detection-global-attention.git
cd salt-dome-detection-global-attention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Organize your seismic data as follows:

```
data/
├── pretrain/
│   ├── kerry3d.npz       # Kerry-3D (Taranaki Basin)
│   ├── opunake3d.npz     # Opunake-3D (Taranaki Basin)
│   └── kahu3d.npz        # Kahu-3D (Taranaki Basin)
├── train/
│   ├── input.npz         # F3 block seismic amplitude volume
│   └── mask.npz          # Zechstein salt binary labels
└── test/
    ├── input.npz
    └── mask.npz
```

**Data Format**:
- `.npz` or `.npy` files
- 3D volumes as `numpy.ndarray`
- Masks: binary (0 = background, 1 = salt)
- Patch size: 64³ with stride 32 (50% overlap)
- Train/validation split: 70/30 (western F3 partition)

**Dataset**: Netherlands F3 block with Zechstein salt annotations from [Alaudah et al. (2019)](https://library.seg.org/doi/10.1190/geo2018-0249.1).

### Training

#### Stage 1: Self-Supervised Pretext Training

```bash
python pretrain_denoising.py \
    --data_dir data/pretrain \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-3 \
    --output_dir checkpoints/pretrain
```

#### Stage 2: Discriminative Transfer Learning

```bash
python train_transfer_learning.py \
    --pretrained_model checkpoints/pretrain/model_epoch_49.pt \
    --input_data data/train/input.npz \
    --mask_data data/train/mask.npz \
    --epochs 100 \
    --base_lr 5e-4 \
    --llrd_decay 0.95 \
    --unfreeze_schedule 0 10 20 30 \
    --batch_size 2 \
    --output_dir checkpoints/finetune
```

The unfreezing schedule unfolds as:
- **Epoch 0**: Decoder + output layers
- **Epoch 10**: Bridge (bottleneck)
- **Epoch 20**: Upper encoder
- **Epoch 30**: Deep encoder

### Inference

```bash
python inference.py \
    --model checkpoints/finetune/best_salt_segmentation_model.pt \
    --input_volume data/test/input.npz \
    --output predictions/output.npz \
    --cube_size 64 \
    --overlap 32
```

## 📈 Training Configuration

### Transfer Learning Techniques

1. **Layer-wise Learning Rate Decay (LLRD)**
   - Base LR: 5e-4 (output/top layers)
   - Decay factor: 0.95 per layer group
   - Preserves low-level noise-discrimination features from pretext training

2. **Gradual Unfreezing**
   - Schedule: [Epoch 0, 10, 20, 30]
   - Progressive adaptation from decoder toward encoder
   - Mitigates catastrophic forgetting of denoising representations

3. **Slanted Triangular LR Schedule (STLR)**
   - Warm-up: 10% of training steps
   - Peak-to-min ratio: 32:1
   - Smooth convergence trajectory for fine-tuning

4. **Unified Focal Loss**
   - Focal Tversky Loss (α=0.3, β=0.7, γ=1.33) — penalizes false negatives
   - Focal Loss (α=0.25, γ=2.0)
   - Balanced weight: λ=0.5
   - Designed for extreme class imbalance (~1.48% salt voxels)

5. **Early Stopping**
   - Patience: 20 epochs
   - Monitor: validation Dice coefficient

## 📁 Repository Structure

```
salt-dome-detection-global-attention/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── salt_dome_segmentation.py       # Main implementation
├── train_transfer_learning.py      # Fine-tuning script
├── pretrain_denoising.py           # Pretext training
├── inference.py                    # Model inference
├── models/
│   ├── __init__.py
│   ├── attention_unet.py          # 3D Global Attention U-Net
│   └── losses.py                  # Unified Focal Loss
├── utils/
│   ├── __init__.py
│   ├── dataset.py                 # SeismicSegmentationDataset
│   ├── metrics.py                 # Evaluation metrics
│   ├── schedulers.py              # STLR, LLRD
│   └── data_loading.py            # Data utilities
├── configs/
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── notebooks/
│   ├── visualization.ipynb
│   └── results_analysis.ipynb
└── tests/
    ├── test_model.py
    └── test_dataset.py
```

## 🔧 Advanced Usage

### Custom Configuration

```yaml
# config.yaml
model:
  in_channels: 1
  out_channels: 1
  base_channels: 16
  attention_heads: 8

training:
  epochs: 100
  batch_size: 2
  base_lr: 5e-4
  llrd_decay: 0.95
  weight_decay: 1e-4
  patience: 20

loss:
  lambda_param: 0.5
  tversky_alpha: 0.3
  tversky_beta: 0.7
  tversky_gamma: 1.33
  focal_alpha: 0.25
  focal_gamma: 2.0

unfreezing:
  schedule: [0, 10, 20, 30]
```

Run with config:

```bash
python train_transfer_learning.py --config config.yaml
```

### TensorBoard Monitoring

```bash
tensorboard --logdir runs/
```

Access at: `http://localhost:6006`

### Export to ONNX

```bash
python export_onnx.py \
    --model checkpoints/best_salt_segmentation_model.pt \
    --output model.onnx \
    --opset_version 14
```

## 📊 Evaluation Metrics

All metrics are computed at voxel level (threshold τ = 0.5, smoothing ε = 1e-8):

- **Dice Coefficient**: Region overlap measure
- **IoU (Jaccard Index)**: Intersection over union
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **F2 Score**: Recall-weighted F-score
- **MCC**: Matthews Correlation Coefficient *(primary metric for class imbalance)*
- **Balanced Accuracy**: Average of sensitivity and specificity
- **Cohen's Kappa**: Agreement beyond chance

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{mahzad2026salt,
  title={Salt Dome Detection in Seismic Data Using a True 3D Global Attention
         Convolutional Network with Self-Supervised Denoising Pretext Training},
  author={Mahzad, Matin and Bagheri, Majid},
  journal={[Scientific Reports]},
  year={2026}
}

@article{mahzad2026denoising,
  title={Self-Supervised Denoising of Seismic Data Using a True 3D Global Attention Convolutional Network},
  author={Mahzad, Matin and Mehrabi, Alireza and Bagheri, Majid},
  journal={Arabian Journal for Science and Engineering},
  year={2026},
  doi={10.1007/s13369-025-10974-5}
}
```

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Matin Mahzad** — matinmahzad@yahoo.com

**Project Link**: [https://github.com/Matin-Mahzad/salt-dome-detection-global-attention](https://github.com/Matin-Mahzad/salt-dome-detection-global-attention)

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Alaudah et al. (2019) for the Netherlands F3 block dataset and Zechstein salt annotations
- Scientific computing community for open-source tools

## 📚 Related Work

This work builds upon and extends:

- **Mahzad, M. & Bagheri, M. (2026)**. Fault Detection in Seismic Data Using a True 3D Global Attention Convolutional Network with Self-Supervised Denoising Pretext Training. *(fault segmentation counterpart to this work)*
- **Mahzad, M., Mehrabi, A., & Bagheri, M. (2026)**. Self-Supervised Denoising of Seismic Data Using a True 3D Global Attention Convolutional Network. *Arabian Journal for Science and Engineering*. [https://doi.org/10.1007/s13369-025-10974-5](https://doi.org/10.1007/s13369-025-10974-5)
- **Alaudah, Y., Michałowicz, P., Alfarraj, M., & AlRegib, G. (2019)**. A machine-learning benchmark for facies classification. *Geophysics, 84*(2), WA175–WA187. *(F3 block dataset)*

---

**Note**: For questions, issues, or feature requests, please open an issue on GitHub.
