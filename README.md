# Can XAI Methods Reliably Identify What a Model Learned?

> **CS 667 — Deep Learning Course Project · Spring 2026 · IIT Gandhinagar**

A systematic comparative evaluation of four Explainable AI (XAI) methods across four neural architectures, assessed on five quantitative metrics with statistical testing, sanity checks, and qualitative heatmap analysis.

---

## Research Question

> *Can existing XAI methods reliably identify what a model actually learned, or do they reflect dataset bias and architecture artefacts?*

**Short answer:** Partially — and the failures are predictable. IntGrad and LIME together form the most reliable combination, but both are shaped by the model's architecture. Interpretable-by-design architectures (BagNet-33) improve reliability more robustly than method selection alone.

---

## Authors

| Name | Email |
|---|---|
| Sankalp Sunil Turankar | sankalp.turankar@iitgn.ac.in |
| Parv Thacker | parv.thacker@iitgn.ac.in |
| Parth Dangi | parth.dangi@iitgn.ac.in |

---

## Project Overview

We evaluate **4 XAI methods** on **4 neural architectures** using **5 quantitative metrics** across **20 Imagenette validation images** (2 per class, 10 classes), producing 320 total attribution evaluations.

### Models

| Model | Architecture | Top-1 Accuracy |
|---|---|---|
| VGG-16 | Sequential CNN, 16 layers, no skip connections | 71.6% |
| ResNet-50 | Residual CNN, 50 layers, skip connections | 76.1% |
| ViT-B16 | Vision Transformer, patch tokens, global attention | 81.1% |
| BagNet-33 | Local-RF CNN, 33×33 receptive field, patch-additive logits | ~70% |

### XAI Methods

| Method | Type | Key Property |
|---|---|---|
| GradCAM | Gradient-based | Reads CNN spatial feature maps; incompatible with ViT |
| IntGrad | Gradient-based | Path-integrated; satisfies completeness axiom |
| LIME | Perturbation-based | Model-agnostic; superpixel-level attribution |
| KernelSHAP | Game-theoretic | Shapley values on superpixels |

### Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Confidence drop after removing top-k% attributed pixels (k = 10, 20, 30%) |
| **AOPC** | 10-step deletion curve (5%–50%); more statistically stable than faithfulness |
| **Sufficiency** | Confidence retained when keeping only top-k% pixels |
| **Sparsity** | Fraction of pixels above attribution threshold 0.5; lower = more focused |
| **Stability** | Cosine similarity across 10 noise-perturbed inputs (σ = 0.05) |

Additional analyses: **ANOVA** (method vs architecture effects), **Pearson correlation** (metric independence), **Spearman r** (cross-architecture attribution agreement), and **cascading parameter randomization** sanity checks.

---

## Key Findings

### 1. Method choice dominates architecture
ANOVA F-statistic by XAI method (21.50) is **5.8× higher** than by architecture (3.70). Choosing IntGrad over KernelSHAP improves faithfulness by up to **10×** regardless of which model is used.

### 2. XAI faithfully exposes dataset bias
Three classes show confirmed spurious correlations across all CNN architectures:
- **Tench → angler** (model learned fishing context, not fish morphology)
- **Chain saw → worker in safety gear** (model learned the person, not the tool)
- **Gas pump → payment-related text** (model learned the signage, not the pump)

All are high-confidence predictions (≥ 99%) with high faithfulness — technically correct explanations of wrong learning.

### 3. GradCAM is architecturally incompatible with ViT-B16
GradCAM produces **uniform zero maps** across all 20 test images on ViT-B16.
Sparsity = 0, Stability = 0, Sanity Pearson r = NaN. This is a mathematical incompatibility — not a bug.

### 4. BagNet-33 achieves the best XAI reliability
- **IntGrad faithfulness: 0.701** — highest in the study (15% above VGG-16 IntGrad)
- **GradCAM sparsity: 0.048** — near IntGrad-level precision (vs 0.772 for VGG-16)
- Correctly identifies fish body in tench image where both CNNs attribute to the angler

### 5. All methods fail under model uncertainty
Low-confidence predictions (< 0.90) produce near-zero or negative faithfulness across all methods and architectures. Explanations are least reliable exactly when they are most needed.

### 6. Metric framework is non-redundant
9 of 10 metric pairs are genuinely independent (|r| < 0.30). Only Faithfulness ↔ AOPC is redundant (r = 0.967). Use AOPC as the primary deletion metric.

### Final Remark
> Out of all models, **IntGrad and LIME can be used together** to explain the contribution of each feature, but their results can be influenced by the model's architecture.

---

## Quantitative Results

### Faithfulness (mean ± std)

| Architecture | GradCAM | IntGrad | LIME | KernelSHAP |
|---|---|---|---|---|
| VGG-16 | 0.367 ± 0.266 | **0.610 ± 0.380** | 0.338 ± 0.310 | 0.130 ± 0.235 |
| ResNet-50 | 0.404 ± 0.305 | **0.559 ± 0.360** | 0.309 ± 0.341 | 0.048 ± 0.113 |
| ViT-B16 | 0.093 ± 0.075 | **0.428 ± 0.290** | 0.247 ± 0.245 | 0.148 ± 0.200 |
| BagNet-33 | 0.550 | **0.701** | 0.360 | 0.263 |

### Stability

| Architecture | GradCAM | IntGrad | LIME | KernelSHAP |
|---|---|---|---|---|
| VGG-16 | **0.996** | 0.812 | 0.853 | 0.545 |
| ResNet-50 | **0.997** | 0.769 | 0.876 | 0.604 |
| ViT-B16 | 0.000* | **0.898** | 0.879 | 0.725 |
| BagNet-33 | **0.983** | 0.759 | 0.850 | 0.156 |

*Zero maps → undefined cosine similarity, not genuine instability.

### ANOVA F-Statistics

| Factor | Faithfulness | AOPC | Sufficiency | Sparsity | Stability |
|---|---|---|---|---|---|
| By Method | 21.50*** | 18.45*** | 4.85** | 52.32*** | 9.92*** |
| By Architecture | 3.70* | 2.78 | 5.56** | 4.25* | 9.70*** |

`***p<0.001  **p<0.01  *p<0.05`

---

## Repository Structure

```
DL_project/
│
├── DL_Project_final (1).py     # Full evaluation pipeline
├── xai_analysis_report.xlsx    # All quantitative results (10 sheets)
├── bagnet.zip                  # BagNet-33 heatmaps + Excel results
├── Pitch deck_DL (1).pptx      # Final presentation (19 slides)
│
└── xai_heatmaps/               # Attribution maps (60 images)
    ├── img01_vgg16.png         # Format: img{NN}_{model}.png
    ├── img01_resnet50.png      # NN = 01–20, model = vgg16 /
    ├── img01_vitb16.png        #   resnet50 / vitb16 / bagnet33
    └── ...                     # 20 images × 3 models × 4 methods
```

### Image–Class Mapping (Excel)

| Image # | Class | Image # | Class |
|---|---|---|---|
| img01–02 | tench | img11–12 | French horn |
| img03–04 | English springer | img13–14 | garbage truck |
| img05–06 | cassette player | img15–16 | gas pump |
| img07–08 | chain saw | img17–18 | golf ball |
| img09–10 | church | img19–20 | parachute |

---

## Setup and Usage

### Requirements

```bash
pip install torch torchvision captum shap lime scikit-image \
            pandas openpyxl matplotlib seaborn scipy numpy tqdm
```

> **CUDA recommended.** The pipeline runs on CPU but is significantly faster on GPU.

### Dataset

The pipeline automatically downloads Imagenette from the fast.ai CDN on first run:

```python
BASE_DIR = "/tmp/imagenette5"   # change as needed
```

### Running the Pipeline

```bash
python "DL_Project_final (1).py"
```

The script will:
1. Download and prepare the Imagenette validation set
2. Load pretrained VGG-16, ResNet-50, ViT-B16, and BagNet-33
3. Run GradCAM, IntGrad, LIME, and KernelSHAP on all 20 images
4. Compute all 5 metrics per attribution map
5. Save heatmaps to `./xai_heatmaps/`
6. Save a checkpoint (`xai_checkpoint.pkl`) every 5 images
7. Export results to `xai_analysis_report.xlsx`

### Resuming from Checkpoint

If the run is interrupted, resume by loading the checkpoint at the top of the script:

```python
import pickle, collections

ck = pickle.load(open("xai_checkpoint.pkl", "rb"))
results       = collections.defaultdict(list, ck["results"])
sanity_results = collections.defaultdict(list, ck["sanity_results"])
stored_attrs  = ck["stored_attrs"]
heatmap_files = ck["heatmap_files"]
```

---

## Implementation Notes

### ViT-B16 GradCAM
Standard GradCAM does not work on Vision Transformers. We implement a manual hook-based variant that:
1. Hooks the final encoder layer (`model.encoder.layers[-1]`)
2. Strips the CLS token from the 197-token output
3. Reshapes the 196 patch tokens to a 14×14 spatial grid
4. Bilinearly upsamples to 224×224

Despite this adaptation, GradCAM produces zero maps on ViT-B16 across all images due to a fundamental incompatibility between the ReLU thresholding in GradCAM and the attention-based token representations. This failure is documented as a key finding of the study.

### KernelSHAP Memory Management
KernelSHAP with 750 samples and 50 superpixels can exceed GPU memory. The prediction function is chunked into batches of 32:

```python
CHUNK = 32
def predict_fn(masked_inputs):
    ...  # batched GPU inference
```

### BagNet-33
BagNet results are generated in a separate run using the same pipeline with `bagnet.zip` containing the output heatmaps and `xai_analysis_report.xlsx` with BagNet-specific results.

---

## Dataset

**Imagenette** — a 10-class subset of ImageNet curated by fast.ai.

- Source: https://github.com/fastai/imagenette
- 10 classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute
- Input size: 224×224, normalised with ImageNet mean/std
- Evaluation: 20 validation images, stratified (2 per class)

---

## Limitations

- **Scale:** 20 images supports comparative conclusions but not per-class statistical significance
- **Hyperparameters:** IntGrad `n_steps=75` (standard: 300); LIME/SHAP `n_samples=750` (standard: 10,000). Results are comparative, not absolute
- **Single baseline:** IntGrad uses a zero baseline only
- **ViT XAI alternatives:** Attention rollout and DINO self-attention are architecturally appropriate for ViT but not evaluated here
- **BagNet accuracy tradeoff:** BagNet-33 has lower top-1 accuracy than ResNet-50 on full ImageNet

---

## Citation

If you use this work, please cite:

```
@misc{turankar2026xai,
  title   = {Can XAI Methods Reliably Identify What a Model Actually Learned,
             or Do They Reflect Dataset Bias and Architecture Artefacts?},
  author  = {Sankalp Sunil Turankar and Parv Thacker and Parth Dangi},
  year    = {2026},
  school  = {Indian Institute of Technology Gandhinagar},
  note    = {CS 667 Deep Learning Course Project, Spring 2026},
  url     = {https://github.com/sankalp567/DL_project}
}
```

---

## Acknowledgements

Project supervised by **Prof. Yogesh Kumar Meena**, HAIX Lab, IIT Gandhinagar.

Built with: PyTorch · Captum · SHAP · LIME · scikit-image · matplotlib · pandas
