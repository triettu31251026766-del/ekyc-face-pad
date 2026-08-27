# Technical Documentation
## Face Presentation Attack Detection (PAD) for Facial eKYC

> **Purpose:** This document is the implementation specification for the student project.
>
> It is written for both students and AI coding assistants.  
> The goal is to make the codebase easy to build incrementally, test, reproduce, and extend.
>
> **Project flow:**
>
> `Dataset → Preprocessing → PAD Baseline → Clean Evaluation → Quality Degradation → Robustness Training → Comparison → Report`

---

# 1. Project Objective

Build a Face Presentation Attack Detection (PAD) pipeline for facial eKYC.

The system must:

1. Load a public face anti-spoofing dataset.
2. Represent each sample as:
   - `bona_fide`
   - `spoof`
3. Train a lightweight image/video PAD model.
4. Evaluate the model on clean data.
5. Create controlled input-quality degradation:
   - JPEG compression
   - resolution reduction
   - blur
   - noise
   - brightness/illumination changes
6. Measure how performance changes as input quality decreases.
7. Train a robustness-enhanced model using controlled quality augmentation.
8. Compare the baseline and robustness-enhanced models under the same evaluation protocol.
9. Save all experiment configurations and results.
10. Generate tables/figures that can be used in the project report.

This is a **course project implementation**, not a production eKYC system.

---

# 2. Implementation Philosophy

The project must be developed in small, testable stages.

Do **not** begin by implementing the complete system.

Required order:

```text
STEP 1
Environment
   ↓
STEP 2
Dataset loading
   ↓
STEP 3
Dataset inspection
   ↓
STEP 4
Preprocessing
   ↓
STEP 5
Model
   ↓
STEP 6
Training
   ↓
STEP 7
Clean evaluation
   ↓
STEP 8
Quality degradation
   ↓
STEP 9
Degraded evaluation
   ↓
STEP 10
Robustness training
   ↓
STEP 11
Comparison
   ↓
STEP 12
Ablation
   ↓
STEP 13
Figures / tables
```

Never skip directly to Step 10.

---

# 3. Recommended Technology Stack

## Required

- Python 3.11+
- PyTorch
- torchvision
- pandas
- NumPy
- scikit-learn
- Pillow
- PyYAML
- Matplotlib

## Optional

- OpenCV
- tqdm
- seaborn
- Jupyter

Use Jupyter only for exploration and visualization.

The actual project pipeline must live in `src/` and `experiments/`.

---

# 4. Repository Structure

```text
ekyc-face-pad/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── clean.yaml
│   ├── degradation_jpeg.yaml
│   ├── degradation_resize.yaml
│   ├── degradation_blur.yaml
│   ├── degradation_noise.yaml
│   └── robustness.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── degradation.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── inference.py
│   ├── video.py
│   ├── robustness.py
│   ├── reproducibility.py
│   └── utils.py
│
├── experiments/
│   ├── train_baseline.py
│   ├── eval_clean.py
│   ├── eval_degradation.py
│   ├── train_robust.py
│   ├── compare_models.py
│   ├── ablation.py
│   └── run_all.py
│
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   └── 02_result_analysis.ipynb
│
└── tests/
    ├── test_data.py
    ├── test_dataset.py
    ├── test_degradation.py
    ├── test_model.py
    ├── test_metrics.py
    └── test_reproducibility.py
```

---

# 5. Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Load and validate YAML configuration |
| `data.py` | Dataset discovery, metadata, train/test split |
| `dataset.py` | PyTorch Dataset/DataLoader |
| `transforms.py` | Normal preprocessing |
| `degradation.py` | Controlled image-quality degradation |
| `model.py` | PAD model definitions |
| `train.py` | Generic training loop |
| `evaluate.py` | Evaluation pipeline |
| `metrics.py` | F1, AUC, APCER, BPCER, ACER |
| `inference.py` | Single image/frame prediction |
| `video.py` | Video frame sampling and aggregation |
| `robustness.py` | Quality augmentation / robust training |
| `reproducibility.py` | Random seed and environment information |
| `utils.py` | Shared utility functions |
| `experiments/` | Experiment orchestration only |

### Separation rule

Do not mix responsibilities.

Examples:

- `model.py` must not implement JPEG compression.
- `degradation.py` must not calculate F1.
- `metrics.py` must not train the model.
- `train.py` must not decide which experiment to run.
- `experiments/` may call modules, but core functionality should remain reusable.

---

# 6. Dataset Contract

The initial recommended dataset is **CelebA-Spoof**.

The implementation must not hard-code assumptions that only work for one directory layout.

Create a normalized internal representation:

```python
Sample(
    path="...",
    label=0,
    subject_id="...",
    attack_type="...",
    metadata={...}
)
```

Recommended label convention:

```text
0 = bona_fide
1 = spoof
```

Use this convention everywhere.

Do not have one module use:

```text
0 = spoof
1 = bona_fide
```

while another uses the opposite.

---

# 7. `src/data.py`

## Responsibilities

- Find dataset files.
- Read metadata/annotations.
- Validate required fields.
- Build normalized sample records.
- Create train/validation/test splits.
- Save split information.

## Recommended API

```python
def discover_dataset(root: str):
    ...

def load_metadata(root: str):
    ...

def build_samples(metadata):
    ...

def create_splits(
    samples,
    seed: int,
):
    ...

def save_splits(splits, path: str):
    ...

def load_splits(path: str):
    ...
```

## Important rule: subject separation

If the dataset provides subject IDs, prefer subject-disjoint splitting.

Do not randomly put images from the same subject into both training and test sets if the project protocol is intended to measure generalization to unseen subjects.

The split strategy must be written to the experiment result.

---

# 8. `src/dataset.py`

Implement a PyTorch Dataset.

Recommended API:

```python
class PADDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        samples,
        transform=None,
    ):
        ...

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...
```

Expected return:

```python
{
    "image": image_tensor,
    "label": label,
    "path": path,
}
```

Optional metadata:

```python
{
    "subject_id": subject_id,
    "attack_type": attack_type,
}
```

Do not return different dictionary structures between training and evaluation.

---

# 9. `src/transforms.py`

Normal preprocessing should be separate from degradation.

Example:

```text
image
 ↓
RGB
 ↓
resize 224×224
 ↓
tensor
 ↓
normalize
```

Recommended API:

```python
def build_train_transform(config):
    ...

def build_eval_transform(config):
    ...
```

The clean evaluation transform must not accidentally contain random augmentation.

---

# 10. `src/degradation.py`

This module is central to the project.

It must implement controlled degradation independently of the model.

## Required functions

```python
def jpeg_compression(image, quality: int):
    ...

def resize_degradation(image, scale: float):
    ...

def gaussian_blur(image, kernel_size: int, sigma: float):
    ...

def gaussian_noise(image, std: float):
    ...

def brightness_adjustment(image, factor: float):
    ...
```

## Unified API

Recommended:

```python
def apply_degradation(
    image,
    degradation_name: str,
    severity,
):
    ...
```

Example:

```python
apply_degradation(
    image,
    degradation_name="jpeg",
    severity=50,
)
```

---

# 11. Degradation Severity

Do not use vague names only.

Bad:

```yaml
severity: strong
```

Better:

```yaml
degradation:
  name: jpeg
  quality: 50
```

or:

```yaml
degradation:
  name: resize
  scale: 0.5
```

or:

```yaml
degradation:
  name: blur
  kernel_size: 7
  sigma: 2.0
```

Every degradation must be reproducible.

---

# 12. Degradation Rules

## JPEG

Example levels:

```text
quality = 90
quality = 70
quality = 50
quality = 30
```

## Resolution

Example:

```text
scale = 1.00
scale = 0.75
scale = 0.50
scale = 0.25
```

## Blur

Example:

```text
small
medium
strong
```

Internally these must map to explicit parameters.

## Noise

Use an explicit standard deviation.

## Brightness

Use an explicit multiplicative factor.

---

# 13. Do Not Randomize Evaluation Degradation

Training may use random augmentation.

Evaluation must use deterministic degradation.

Bad:

```python
test_image = random_blur(test_image)
```

Good:

```python
test_image = gaussian_blur(
    test_image,
    kernel_size=7,
    sigma=2.0,
)
```

The same evaluation configuration must produce the same transformed sample.

---

# 14. `src/model.py`

Start with a lightweight model.

Recommended first baseline:

```text
MobileNetV2
```

Alternative:

```text
MobileNetV3
```

A small custom CNN can be used as a debugging baseline.

## Recommended API

```python
def build_model(
    model_name: str,
    num_classes: int = 1,
):
    ...
```

For binary classification, use one output logit.

Example:

```python
logit = model(image)
probability = torch.sigmoid(logit)
```

Use:

```python
torch.nn.BCEWithLogitsLoss
```

rather than applying sigmoid before the loss.

---

# 15. Class Imbalance

Face anti-spoofing datasets may have different class distributions depending on the selected split.

Before training, calculate:

```text
num_bona_fide
num_spoof
spoof_ratio
```

Do not blindly add class weighting.

First inspect the distribution.

If class weighting is used, put it in configuration:

```yaml
loss:
  name: bce_with_logits
  use_pos_weight: true
```

The same training policy must be used when comparing baseline and robustness models unless the experiment specifically studies the loss.

---

# 16. `src/train.py`

Training must be model-agnostic.

Recommended functions:

```python
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
):
    ...

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs,
    device,
):
    ...
```

Return structured information:

```python
{
    "epoch": epoch,
    "train_loss": ...,
    "val_loss": ...,
}
```

Do not write experiment-specific filenames inside `train.py`.

---

# 17. `src/metrics.py`

Implement all metrics in one place.

Required:

```python
def classification_metrics(
    y_true,
    y_prob,
    threshold=0.5,
):
    ...
```

Return:

```python
{
    "accuracy": ...,
    "precision": ...,
    "recall": ...,
    "f1": ...,
    "roc_auc": ...,
    "pr_auc": ...
}
```

PAD metrics:

```python
def apcer(y_true, y_pred):
    ...

def bpcer(y_true, y_pred):
    ...

def acer(y_true, y_pred):
    ...
```

The exact positive/negative convention must be documented in code.

Recommended convention:

```text
positive = spoof
negative = bona_fide
```

Then:

```text
APCER:
spoof predicted as bona_fide

BPCER:
bona_fide predicted as spoof
```

---

# 18. `src/evaluate.py`

Evaluation must be deterministic.

Recommended API:

```python
def evaluate_model(
    model,
    dataloader,
    device,
    threshold=0.5,
):
    ...
```

Return:

```python
{
    "metrics": {...},
    "predictions": [...],
    "probabilities": [...],
    "labels": [...],
}
```

Optionally save raw predictions:

```text
path
label
probability
prediction
```

This makes later error analysis possible.

---

# 19. `src/inference.py`

Provide a small inference API:

```python
def predict_image(
    model,
    image_path,
    transform,
    device,
):
    ...
```

Return:

```python
{
    "probability_spoof": ...,
    "prediction": "spoof" | "bona_fide",
}
```

Do not create a web server for the first version.

---

# 20. `src/video.py`

Use only if the selected dataset requires video processing.

Recommended pipeline:

```text
video
 ↓
sample N frames
 ↓
face crop
 ↓
preprocess
 ↓
PAD model
 ↓
frame probabilities
 ↓
aggregate
 ↓
video probability
```

Recommended functions:

```python
def sample_frames(video_path, num_frames):
    ...

def predict_frames(model, frames):
    ...

def aggregate_frame_scores(scores, method="mean"):
    ...
```

Start with:

```python
method="mean"
```

Do not add a complex temporal model until the frame-level pipeline works.

---

# 21. `src/robustness.py`

The first robustness implementation should be simple.

Recommended concept:

```text
Original training sample
       │
       ├── clean
       ├── JPEG degraded
       ├── resized
       ├── blurred
       └── noisy
              │
              ▼
       lightweight PAD model
```

The module should decide which quality transformations are enabled based on config.

Recommended API:

```python
def build_robustness_transform(config):
    ...

def apply_training_quality_augmentation(
    image,
    config,
):
    ...
```

Do not duplicate degradation implementations here.

`robustness.py` must call functions from `degradation.py`.

---

# 22. Clean Baseline

The first experiment must be:

```text
Dataset
 ↓
Clean preprocessing
 ↓
Lightweight model
 ↓
Training
 ↓
Clean test
 ↓
Metrics
```

Example:

```bash
python experiments/train_baseline.py \
    --config configs/clean.yaml
```

Expected output:

```text
results/raw/E01_baseline_seed42.json
results/raw/E01_baseline_seed42.csv
```

---

# 23. Degradation Evaluation

After the clean baseline works:

```text
same trained model
       ↓
clean test
       ↓
JPEG 90
       ↓
JPEG 70
       ↓
JPEG 50
       ↓
JPEG 30
```

Repeat for:

```text
resize
blur
noise
brightness
```

Do not retrain the baseline for each degradation.

The purpose is to measure how the same model reacts to different input conditions.

---

# 24. Robust Training

After baseline degradation results exist:

```text
Quality augmentation
       ↓
Train model
       ↓
Clean test
       ↓
Degraded test
       ↓
Compare against baseline
```

The comparison must keep these fixed where possible:

```text
dataset split
model architecture
optimizer
learning rate
epochs
batch size
test set
threshold
random seed
```

The intended experimental variable is the robustness training strategy.

---

# 25. Experiment Configuration

## `configs/base.yaml`

```yaml
seed: 42

dataset:
  name: celeba_spoof
  root: data/raw/celeba_spoof

split:
  strategy: subject_disjoint

model:
  name: mobilenet_v2
  image_size: 224

training:
  epochs: 20
  batch_size: 64
  learning_rate: 0.0001
  weight_decay: 0.00001

loss:
  name: bce_with_logits
  use_pos_weight: false

evaluation:
  threshold: 0.5

device:
  name: auto
```

---

# 26. Example Degradation Config

```yaml
seed: 42

degradation:
  name: jpeg
  quality: 50
```

Another:

```yaml
degradation:
  name: resize
  scale: 0.5
```

Another:

```yaml
degradation:
  name: blur
  kernel_size: 7
  sigma: 2.0
```

---

# 27. Example Robustness Config

```yaml
seed: 42

robustness:
  enabled: true

  augmentations:
    jpeg:
      enabled: true
      quality_range: [50, 90]

    resize:
      enabled: true
      scale_range: [0.5, 1.0]

    blur:
      enabled: true
      sigma_range: [0.5, 2.0]

    noise:
      enabled: true
      std_range: [0.005, 0.03]

    brightness:
      enabled: true
      factor_range: [0.7, 1.3]
```

The exact ranges are experiment parameters and must be recorded with the result.

---

# 28. Experiment IDs

Every run gets a unique ID.

Examples:

```text
E01_baseline_seed42
E02_jpeg90_seed42
E03_jpeg70_seed42
E04_jpeg50_seed42
E05_resize50_seed42
E06_blur_medium_seed42
E07_robust_seed42
```

Use IDs consistently in filenames.

---

# 29. Result Schema

Every result should contain at least:

```text
experiment_id
seed
dataset
split_strategy
model
training_mode
degradation_name
degradation_parameters
threshold
accuracy
precision
recall
f1
roc_auc
pr_auc
apcer
bpcer
acer
runtime_seconds
```

Optional:

```text
parameter_count
model_size_mb
inference_latency_ms
```

---

# 30. Result File Format

Save both CSV and JSON.

Example:

```text
results/raw/E01_baseline_seed42.csv
results/raw/E01_baseline_seed42.json
```

A JSON record may look like:

```json
{
  "experiment_id": "E01_baseline_seed42",
  "seed": 42,
  "model": "mobilenet_v2",
  "training_mode": "clean",
  "degradation": {
    "name": "none"
  },
  "metrics": {
    "f1": 0.0,
    "roc_auc": 0.0,
    "apcer": 0.0,
    "bpcer": 0.0,
    "acer": 0.0
  }
}
```

The values above are placeholders only. Never write fake results into the repository.

---

# 31. Experiment Scripts

## `train_baseline.py`

Responsibilities:

```text
load config
→ set seed
→ load dataset
→ load splits
→ build transforms
→ build model
→ train
→ save checkpoint
→ evaluate clean test
→ save results
```

---

## `eval_clean.py`

Responsibilities:

```text
load checkpoint
→ load clean test set
→ evaluate
→ save metrics
```

---

## `eval_degradation.py`

Responsibilities:

```text
load checkpoint
→ load test set
→ apply deterministic degradation
→ evaluate
→ save metrics
```

Arguments should include:

```text
--config
--checkpoint
```

---

## `train_robust.py`

Responsibilities:

```text
load config
→ enable robustness augmentation
→ train model
→ save checkpoint
→ evaluate clean test
→ save results
```

---

## `compare_models.py`

Responsibilities:

```text
load saved result files
→ align experiments
→ create comparison table
→ save CSV
→ generate figures
```

Do not retrain models inside `compare_models.py`.

---

## `ablation.py`

Responsibilities:

```text
run/read individual augmentation variants
→ compare metrics
→ save ablation table
```

---

# 32. Experiment Order

Run exactly this progression.

```text
E01
Clean baseline
        ↓
E02–E06
Degradation stress tests
        ↓
E07
Robust training
        ↓
E08
Robust model under degradation
        ↓
E09+
Ablation
        ↓
Final tables / figures
```

Do not run large grids until E01 succeeds.

---

# 33. Minimum First Milestone

The first successful milestone is intentionally small:

```text
Dataset
 ↓
Train/test split
 ↓
MobileNetV2
 ↓
5–10 epochs
 ↓
Clean test
 ↓
F1 + ROC-AUC
```

The expected output is:

```text
A trained checkpoint
+
A metrics JSON
+
A metrics CSV
```

Only after this works should degradation code be added.

---

# 34. Unit Tests

## `tests/test_data.py`

Test:

- dataset path validation;
- labels are only `0` and `1`;
- train/test have no overlapping subject IDs when using subject-disjoint split;
- every expected sample is assigned.

---

## `tests/test_dataset.py`

Test:

```text
len(dataset)
sample["image"]
sample["label"]
```

Expected:

```text
image → torch.Tensor
label → integer / tensor
```

---

## `tests/test_degradation.py`

Test that:

```text
JPEG output ≠ original
resize output has expected dimensions
blur output has expected dimensions
noise output has expected dimensions
brightness output has expected dimensions
```

Also test that invalid parameters raise clear errors.

---

## `tests/test_model.py`

Test:

```text
input shape
forward pass
output shape
loss calculation
```

Example:

```python
x = torch.randn(2, 3, 224, 224)
y = model(x)

assert y.shape[0] == 2
```

---

## `tests/test_metrics.py`

Use tiny manually verifiable examples.

Test:

- all predictions correct;
- all predictions wrong;
- only bona fide;
- only spoof;
- imbalanced labels.

---

# 35. Synthetic Degradation Test

Before using the real dataset:

```python
from PIL import Image

image = Image.new("RGB", (224, 224), color="gray")
```

Then verify each degradation.

This isolates image-processing bugs from dataset/model bugs.

---

# 36. Reproducibility

Implement:

```python
def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Also save:

```text
Python version
PyTorch version
torchvision version
dataset information
config file
seed
Git commit
device
```

---

# 37. Logging

Each experiment should log:

```text
experiment_id
timestamp
seed
config
model
epoch
train_loss
validation_loss
final_metrics
runtime
```

Do not rely only on terminal output.

---

# 38. Checkpointing

Save:

```text
results/checkpoints/
```

or:

```text
checkpoints/
```

Recommended checkpoint content:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "config": config,
    "seed": seed,
}
```

Do not save only raw model weights if reproducibility matters.

---

# 39. Error Analysis

Save predictions so the team can inspect mistakes.

Recommended CSV:

```text
path
subject_id
attack_type
label
probability_spoof
prediction
correct
```

Then inspect:

```text
false positives
false negatives
```

Useful questions:

```text
Which spoof types are difficult?
Does compression increase false negatives?
Does low resolution increase false positives?
Does robustness training reduce specific errors?
```

---

# 40. Figure Generation

Figures must be generated from saved result files.

Recommended figures:

```text
fig_acer_vs_jpeg_quality.png
fig_f1_vs_resolution.png
fig_f1_baseline_vs_robust.png
fig_apcer_bpcer_comparison.png
fig_ablation.png
```

Example conceptual plot:

```text
Quality
  │
  │\
  │ \
  │  \
  │   \
  │    \
  └────────────
       ↓
   Performance
```

Do not manually type numbers into plots.

---

# 41. Comparison Rules

When comparing baseline vs robust model, keep fixed:

```text
dataset
split
test samples
model architecture
evaluation threshold
metric implementation
```

If the training setup changes, record it.

The comparison must be:

```text
Baseline
vs
Robustness training
```

not:

```text
small model + 10 epochs
vs
large model + 50 epochs + different test set
```

---

# 42. Latency Measurement

If latency is included, use a defined protocol.

Example:

```python
# warm-up
for _ in range(20):
    model(x)

# measured inference
start = time.perf_counter()

for _ in range(100):
    model(x)

end = time.perf_counter()
```

Report:

```text
device
batch size
image size
number of runs
warm-up count
mean latency
```

Do not compare latency measured on different hardware without stating the difference.

---

# 43. AI Coding Agent Rules

This section is mandatory when using an AI coding assistant.

## Rule 1 — Read this document first

Before modifying code, the AI must:

1. Read this document.
2. Inspect the repository tree.
3. Inspect relevant existing files.
4. Identify dependencies.
5. State the planned changes.

---

## Rule 2 — Implement one module at a time

Recommended order:

```text
config.py
 ↓
data.py
 ↓
dataset.py
 ↓
transforms.py
 ↓
model.py
 ↓
train.py
 ↓
metrics.py
 ↓
evaluate.py
 ↓
degradation.py
 ↓
robustness.py
 ↓
video.py
 ↓
experiments/
```

---

## Rule 3 — Do not rewrite unrelated files

If asked:

```text
Implement degradation.py
```

the AI must not rewrite:

```text
model.py
train.py
server.py
README.md
```

unless required for integration.

---

## Rule 4 — Run tests after implementation

Every module implementation should end with:

```text
1. syntax check
2. unit tests
3. small execution test
```

---

## Rule 5 — No fake results

Never invent:

```text
F1
AUC
APCER
BPCER
ACER
latency
```

If the experiment was not executed:

```text
"The result has not been measured yet."
```

---

## Rule 6 — Preserve experiment settings

AI must not silently change:

```text
seed
dataset split
model
learning rate
epochs
batch size
threshold
degradation severity
```

---

# 44. Recommended AI Prompts

## Prompt 1 — Dataset

```text
Read docs/TECHNICAL_DOCUMENTATION.md first.

Implement only src/data.py.

Requirements:
- load the selected face anti-spoofing dataset
- normalize samples to the project label convention
- support subject-disjoint splitting if subject IDs are available
- make the split reproducible
- save/load split information
- add unit tests

Do not implement model training yet.

After implementation:
1. run tests
2. run a small dataset inspection
3. report changed files
4. report assumptions
```

---

## Prompt 2 — Dataset class

```text
Read the technical documentation.

Implement src/dataset.py.

Requirements:
- PyTorch Dataset
- return image, label and path
- use the existing split representation
- support train/eval transforms
- add tests
- do not implement model logic
```

---

## Prompt 3 — Model

```text
Read the technical documentation.

Implement src/model.py.

Requirements:
- support MobileNetV2 first
- binary classification
- one output logit
- expose build_model(...)
- keep architecture configurable
- add forward-pass tests

Do not modify training or experiment scripts.
```

---

## Prompt 4 — Training

```text
Read the technical documentation.

Implement src/train.py.

Requirements:
- generic PyTorch training loop
- optimizer and loss supplied from caller
- train_one_epoch(...)
- train_model(...)
- return structured history
- no experiment-specific file paths
- add tests/smoke test
```

---

## Prompt 5 — Metrics

```text
Read the technical documentation.

Implement src/metrics.py.

Requirements:
- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- APCER
- BPCER
- ACER

Convention:
positive = spoof
negative = bona_fide

Add small deterministic unit tests.
```

---

## Prompt 6 — Degradation

```text
Read the technical documentation.

Implement src/degradation.py only.

Requirements:
- JPEG compression
- resize degradation
- Gaussian blur
- Gaussian noise
- brightness adjustment
- deterministic parameters
- unified apply_degradation(...)
- validation of parameters
- unit tests using a synthetic image

Do not modify model or training code.
```

---

## Prompt 7 — Baseline experiment

```text
Read the technical documentation.

Implement experiments/train_baseline.py.

Pipeline:
dataset → split → transform → MobileNetV2 → training → clean evaluation → save checkpoint/results.

Use configuration from YAML.

Do not implement robustness yet.
```

---

## Prompt 8 — Degradation evaluation

```text
Read the technical documentation.

Implement experiments/eval_degradation.py.

Requirements:
- load an existing baseline checkpoint
- use the same test split
- apply one deterministic degradation from config
- evaluate
- save CSV and JSON
- do not retrain the model
```

---

## Prompt 9 — Robustness training

```text
Read the technical documentation.

Implement src/robustness.py and experiments/train_robust.py.

Requirements:
- reuse degradation.py
- configurable quality augmentation
- keep model architecture unchanged
- keep evaluation protocol unchanged
- save checkpoint and metrics
- add tests
```

---

## Prompt 10 — Comparison

```text
Read the technical documentation.

Implement experiments/compare_models.py.

Requirements:
- read saved result files
- compare baseline and robust model
- generate CSV tables
- generate figures from raw result files
- do not retrain models
- do not manually enter metrics
```

---

# 45. Debugging Order

When something fails, follow this order:

```text
1. Dataset path
        ↓
2. Dataset metadata
        ↓
3. Labels
        ↓
4. Train/test split
        ↓
5. Transform
        ↓
6. Model forward pass
        ↓
7. Loss
        ↓
8. Training
        ↓
9. Clean evaluation
        ↓
10. Degradation
        ↓
11. Robustness training
        ↓
12. Full experiment
```

Do not debug robustness training when clean training is broken.

---

# 46. Common Bugs

## Bug: Accuracy is high but F1 is low

Check:

```text
class distribution
confusion matrix
precision
recall
```

---

## Bug: Model predicts only one class

Check:

```text
label mapping
loss
class distribution
learning rate
```

---

## Bug: Degraded images look identical

Check:

```text
degradation parameters
PIL/OpenCV conversion
image dtype
image range
```

---

## Bug: Robust model looks better only because it trained longer

Keep:

```text
epochs
optimizer
learning rate
batch size
```

consistent.

---

## Bug: Results change every run

Check:

```text
Python seed
NumPy seed
PyTorch seed
DataLoader randomness
split generation
```

---

## Bug: Evaluation numbers cannot be reproduced

Check:

```text
saved split
saved config
saved checkpoint
saved threshold
saved degradation parameters
```

---

# 47. Final Experiment Set

A practical final set is:

| Experiment | Training | Test |
|---|---|---|
| E01 | Clean | Clean |
| E02 | Clean | JPEG 90 |
| E03 | Clean | JPEG 70 |
| E04 | Clean | JPEG 50 |
| E05 | Clean | JPEG 30 |
| E06 | Clean | Resolution 75/50/25% |
| E07 | Clean | Blur levels |
| E08 | Clean | Noise levels |
| E09 | Clean | Brightness levels |
| E10 | Robust augmentation | Clean |
| E11 | Robust augmentation | JPEG |
| E12 | Robust augmentation | Resolution |
| E13 | Robust augmentation | Blur/noise |
| E14 | Robust augmentation | Multiple quality conditions |
| E15+ | Ablation | Selected conditions |

Only add experiments when they provide useful information for the project.

---

# 48. Definition of Done

The codebase is ready for final project experiments when:

- [ ] Dataset loads successfully.
- [ ] Labels are correct and consistent.
- [ ] Train/test split is reproducible.
- [ ] Dataset class works.
- [ ] Clean preprocessing works.
- [ ] MobileNetV2 baseline trains.
- [ ] Clean evaluation works.
- [ ] F1/AUC are calculated.
- [ ] APCER/BPCER/ACER are calculated.
- [ ] JPEG degradation works.
- [ ] Resize degradation works.
- [ ] Blur degradation works.
- [ ] Noise degradation works.
- [ ] Brightness degradation works.
- [ ] Degradation evaluation is deterministic.
- [ ] Robustness augmentation works.
- [ ] Baseline vs robust comparison works.
- [ ] Results are automatically saved.
- [ ] Figures are generated from saved results.
- [ ] Unit tests pass.
- [ ] At least one complete experiment can be reproduced from a config file.

---

# 49. Final Implementation Flow

The entire project should remain understandable as:

```text
DATASET
   ↓
SPLIT
   ↓
PREPROCESSING
   ↓
LIGHTWEIGHT PAD MODEL
   ↓
CLEAN TRAINING
   ↓
CLEAN EVALUATION
   ↓
QUALITY DEGRADATION
   ↓
DEGRADED EVALUATION
   ↓
QUALITY-AUGMENTED TRAINING
   ↓
ROBUST EVALUATION
   ↓
COMPARISON
   ↓
ABLATION
   ↓
TABLES / FIGURES
   ↓
PROJECT REPORT
```

If a new piece of code does not fit naturally into this flow, stop and decide where it belongs before implementing it.

The goal is a **small, understandable, reproducible engineering project**, not a large research framework.
