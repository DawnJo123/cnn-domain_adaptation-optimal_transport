# Optimal Transport for CNN-Based Domain Adaptation

## Overview

This project investigates **feature-level Optimal Transport (OT) for unsupervised domain adaptation** using frozen CNN features from the Office-31 dataset.

The initial experiment adapts an Amazon source domain to a Webcam target domain. A pretrained ResNet-18 extracts image representations. Four OT approaches map source training features toward unlabeled target adaptation features. Logistic Regression classifiers are trained on the mapped source features and evaluated on a held-out, labeled target test set.

**Current result:** None of the evaluated OT configurations improved target-domain classification performance over the source-only baseline.

## Research question

Can feature-level Optimal Transport improve the transfer of a classifier trained on labeled Amazon images to the Webcam domain, compared with training on the original source features alone?

The experiment compares:

- Source-only training (baseline)
- Exact Earth Mover's Distance (EMD) transport
- Entropic Sinkhorn transport
- Class-regularized Sinkhorn with LpL1 regularization
- Class-regularized Sinkhorn with L1L2 regularization

## Dataset

**Office-31** is a domain adaptation benchmark containing 31 object categories across three domains.

| Domain | Images |
|---|---:|
| Amazon | 2,817 |
| Webcam | 795 |
| DSLR | 498 |
| **Total** | **4,110** |

This experiment uses:

- **Source domain:** Amazon
- **Target domain:** Webcam
- **Direction:** Amazon → Webcam
- **Classes:** 31

The DSLR domain is not used in this experiment.

## Experimental protocol

### Data splits

The split was generated using seed `42`.

| Split | Domain | Samples | Use |
|---|---|---:|---|
| Source train | Amazon | 2,253 | Feature-level adaptation and classifier training |
| Source test | Amazon | 564 | Source-domain baseline evaluation |
| Target adaptation | Webcam | 397 | Unlabeled target samples used to estimate OT transport |
| Target test | Webcam | 398 | Held-out final target-domain evaluation |

The Amazon source split is stratified. The Webcam target adaptation/test partition was created by a random half split; it was not stratified. All 31 classes are represented in the target test split.

The split manifest is stored at `experiments/office31_splits.csv`. Labels for the target adaptation split are not included in the manifest. Target test labels are reserved for final evaluation and are not used to fit OT mappings or classifiers.

### Feature extraction

A pretrained **ResNet-18** is used as a frozen feature extractor.

- The final fully connected classification layer is replaced with `nn.Identity()`.
- Each image is represented by a 512-dimensional feature vector.
- The pretrained model is used in evaluation mode, without gradient computation.
- Image preprocessing follows the transforms associated with the pretrained weights.

Extracted features are saved in:

`experiments/features/office31_amazon_webcam_features.pt`

Expected feature shapes:

| Feature set | Shape |
|---|---:|
| `Xs_train` | `(2253, 512)` |
| `Xs_test` | `(564, 512)` |
| `Xt_adapt` | `(397, 512)` |
| `Xt_test` | `(398, 512)` |

The feature file also stores source labels, held-out evaluation labels, class metadata, and the random seed. Target test labels are used only for final evaluation.

### Optimal Transport

The source-to-target cost matrix is computed using squared Euclidean distance:

$$
C_{ij} = \|x_i^s - x_j^t\|_2^2
$$

The source and target empirical distributions use uniform sample weights:

$$
a_i = \frac{1}{n_s}, \qquad b_j = \frac{1}{n_t}
$$

The transport plan is estimated from Amazon source-training features and unlabeled Webcam target-adaptation features. For class-regularized methods, only the source training labels are supplied.

The evaluated methods are:

| Method | Description |
|---|---|
| EMD | Exact OT using `ot.da.EMDTransport` |
| Sinkhorn | Entropically regularized OT using `ot.da.SinkhornTransport` |
| Sinkhorn LpL1 | Class-regularized Sinkhorn using `ot.da.SinkhornLpl1Transport` |
| Sinkhorn L1L2 | Class-regularized Sinkhorn using `ot.da.SinkhornL1l2Transport` |

For Sinkhorn variants, the initial entropy regularization was set to 5% of the mean cost:

`reg_e = 0.05 × mean(C) = 32.8204453315`

The class-regularization values used in this initial experiment were:

- LpL1: `reg_cl = 1.0`
- L1L2: `reg_cl = 2.0`

These are initial experimental settings, not values established through a hyperparameter search.

### Feature mapping and classification

Each fitted transport estimator is used to compute a barycentric mapping of the source training features into the target feature space.

A Logistic Regression classifier is trained separately for each feature set:

- Source-only baseline: original Amazon source-training features.
- OT methods: adapted Amazon source-training features, retaining the original source labels.

Classifier configuration:

| Parameter | Value |
|---|---|
| `C` | `1.0` |
| `solver` | `lbfgs` |
| `max_iter` | `2000` |
| `random_state` | `42` |

All classifiers are evaluated on the **same, unchanged Webcam target test features**.

Metrics:

- Accuracy
- Balanced accuracy
- Macro F1

Balanced accuracy and macro F1 are included because the target test split is not stratified.

## Results

### Target-domain evaluation

The following results are from the held-out Webcam target test set.

| Method | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Source Only | **0.6357** | **0.6604** | **0.6289** |
| EMD | 0.5905 | 0.5952 | 0.5864 |
| Sinkhorn | 0.5226 | 0.5202 | 0.5042 |
| Sinkhorn LpL1 | 0.5251 | 0.5225 | 0.5077 |
| Sinkhorn L1L2 | 0.5251 | 0.5227 | 0.5062 |

### Change relative to the source-only baseline

Values are absolute percentage-point differences from the source-only target-test result.

| Method | Accuracy (pp) | Balanced Accuracy (pp) | Macro F1 (pp) |
|---|---:|---:|---:|
| Source Only | 0.00 | 0.00 | 0.00 |
| EMD | -4.52 | -6.52 | -4.25 |
| Sinkhorn | -11.31 | -14.01 | -12.47 |
| Sinkhorn LpL1 | -11.06 | -13.79 | -12.12 |
| Sinkhorn L1L2 | -11.06 | -13.77 | -12.27 |

### Source-domain baseline

The source-only classifier achieved the following result on the held-out Amazon source test set:

| Dataset | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Amazon Source Test | 0.8369 | 0.8335 | 0.8329 |

The source-only target-test accuracy was 63.57%, compared with 83.69% on the source test set: a difference of **20.12 percentage points**. This indicates a substantial source-to-target performance gap for the baseline classifier.

## Findings and current conclusion

For this Amazon → Webcam split and the tested configurations:

1. **The source-only baseline performed best** on all three target-domain metrics.
2. EMD was the closest OT method to the baseline, but its target accuracy was lower by 4.52 percentage points.
3. The three Sinkhorn variants had lower target performance than EMD and the source-only baseline.
4. LpL1 and L1L2 class regularization produced very similar results to standard Sinkhorn in this experiment. Neither recovered the performance reduction.
5. The source-only classifier showed a 20.12 percentage-point accuracy difference between the Amazon source test set and Webcam target test set.

**Conclusion:** The evaluated feature-level OT configurations did not improve target-domain classification performance in this initial experiment. The results do not establish that Optimal Transport is generally ineffective for domain adaptation. They show that the tested transport objectives and hyperparameter settings, combined with this frozen ResNet-18 representation and barycentric mapping procedure, did not outperform the source-only baseline on this split.

The experiment uses one domain direction and one random split. The findings should therefore be treated as preliminary.

## Limitations and interpretation

- Only Amazon → Webcam is evaluated; the DSLR domain and reverse domain directions remain untested.
- Results are based on one random split (seed 42). Variability across seeds has not yet been measured.
- The target adaptation split is not stratified.
- The Sinkhorn and class-regularization parameters are initial settings, not the outcome of a systematic sensitivity study.
- Target test labels are held out from adaptation and training. They are used for final metric calculation only.
- OT minimizes a transport objective; it does not directly optimize target classification accuracy.
- The current experiment is **post-hoc feature-level adaptation**. The ResNet-18 feature extractor is frozen; it is not trained end-to-end with an OT loss.
- The source test set is used for the source-only baseline. It is not used to evaluate the adapted classifiers, because those classifiers are trained on source features mapped into the target feature space. Comparing them against unadapted source-test features would mix feature spaces.

## Repository structure

```text
cnn-domain-adaptation-optimal-transport/
├── data/
│   └── Office-31/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_domain_adaptation.ipynb
│   └── 04_classifier.ipynb
├── experiments/
│   ├── office31_splits.csv
│   └── features/
│       ├── office31_amazon_webcam_features.pt
│       └── adapted/
│           ├── emd.pt
│           ├── sinkhorn.pt
│           ├── sinkhorn_lpl1.pt
│           └── sinkhorn_l1l2.pt
└── pyproject.toml
```

### Notebook sequence

| Notebook | Purpose |
|---|---|
| `01_dataset_exploration.ipynb` | Inspect Office-31 and create the reproducible split manifest |
| `02_feature_extraction.ipynb` | Extract frozen ResNet-18 features and save feature tensors |
| `03_domain_adaptation.ipynb` | Estimate OT plans, map source features, and save adapted features |
| `04_classifier.ipynb` | Train classifiers and evaluate source-only and adapted representations |

Generated feature tensors and transport plans are experiment artifacts. They are not raw dataset files.

## Environment and dependencies

The experiment was run with:

| Component | Version / configuration |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 |
| Torchvision | 0.21.0+cu124 |
| POT | 0.9.7.post1 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 4 GB |
| Operating system | Windows |
| Notebook environment | VS Code |

The project uses PyTorch/torchvision for feature extraction, POT (Python Optimal Transport) for OT estimators, and scikit-learn for Logistic Regression and evaluation metrics. NumPy, pandas, and Matplotlib are used for numerical work and analysis.

Install the project dependencies from `pyproject.toml` using the project's chosen environment manager. The recorded PyTorch installation uses the CUDA 12.4 wheel index; ensure the installed PyTorch build is compatible with the local NVIDIA driver.

## Reproducibility

- Random seed: `42`
- Split manifest: `experiments/office31_splits.csv`
- Extracted feature artifact: `experiments/features/office31_amazon_webcam_features.pt`
- Adapted feature and transport artifacts: `experiments/features/adapted/`
- Classifier settings are recorded above.
- OT settings are recorded in the corresponding saved experiment artifacts and in Notebook 03.

Run the notebooks in numerical order. The Office-31 dataset must be placed under `data/Office-31/` in the expected directory structure before running dataset exploration and feature extraction.

## Next experiments

The following analyses are planned, not yet completed:

1. Inspect transport-plan concentration and effective numbers of target connections.
2. Analyze barycentric feature displacement and changes in class separation.
3. Examine per-class target performance and confusion matrices.
4. Conduct controlled sensitivity analyses for entropy and class-regularization parameters, without using target test labels for tuning.
5. Repeat the experiment across multiple random seeds.
6. Evaluate additional domain directions, including Webcam → Amazon and experiments involving DSLR.
7. Investigate trainable feature extraction with an OT alignment objective as a separate deep-OT experiment.

## References

- Office-31: Saenko et al., *Adapting Visual Category Models to New Domains*, ECCV 2010.
- Optimal Transport: Peyré and Cuturi, *Computational Optimal Transport*, Foundations and Trends in Machine Learning, 2019.
- Python Optimal Transport (POT): https://pythonot.github.io/
- PyTorch: https://pytorch.org/
- Torchvision: https://pytorch.org/vision/
- scikit-learn: https://scikit-learn.org/
