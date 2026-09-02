# -*- coding: utf-8 -*-
"""
CONTROL BOARD — user-specified paths, toggles, and hyperparameters.

Edit this file (not YAML) before a real training run. Smoke tests and HPO
copy these defaults into a TrainingConfig and override a few fields.

Data and frozen/checkpoint weights live on the E: removable disk. Per-run
reports (curves, logs, freeze decision) still go under reporting/training/
so the repo keeps a paper trail without storing weights in git.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from arth_tools.paths import TRAINING_REPORT_DIR

# ============================================================================
# CONTROL BOARD - USER-SPECIFIED PARAMETERS
# ============================================================================

# --- Removable disk (models + data) ---
# Change the drive letter if the volume mounts elsewhere.
DATA_ROOT = Path(r"E:\ArthAgent")
DICOMDIR_PATH = Path(r"E:\DICOMwrapper")
DICOM_ROOT = Path(r"E:\DICOMwrapper\DICOM\CANON\20250512")

IMAGE_ROOT = DATA_ROOT / "images"
PICKLE_DIR = DATA_ROOT / "pickles"
MANIFEST_DIR = DATA_ROOT / "manifests"
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
FROZEN_DIR = DATA_ROOT / "frozen"

# CSV manifests listing image files AND labels (one row per slice).
MANIFEST_MASTER = MANIFEST_DIR / "manifest_master.csv"
MANIFEST_TRAIN = MANIFEST_DIR / "manifest_train.csv"
MANIFEST_VAL = MANIFEST_DIR / "manifest_val.csv"
MANIFEST_TEST = MANIFEST_DIR / "manifest_test.csv"

# Filename column & label column names in manifests (CSV header).
FILEPATH_COLUMN = "filepath"
LABEL_COLUMN = "label"
PATIENT_ID_COLUMN = "patient_id"

# --- Task / outputs ---
# Loss: 'binary_crossentropy' | 'sparse_categorical_crossentropy' | 'mse'
# categorical_crossentropy expects y_true one-hot encoded (we use sparse instead)
LOSS_NAME = "sparse_categorical_crossentropy"

# Output layer type: 'sigmoid', 'softmax', 'linear'
# The final Dense is linear (logits). Sigmoid / softmax are applied when scoring,
# which is the numerically stable equivalent of Keras' in-layer activations.
OUTPUT_TYPE = "softmax"

# Number of classes for multi-class classification (zero-indexed labels)
NUM_CLASSES = 2

# --- Architecture ---
# 'cnn' / 'tiny_cnn' → explicit VGG-inspired build_cnn_model()
# 'resnet50' / 'microsoft/resnet-50' → Hugging Face ResNet (needs transformers)
ARCHITECTURE_ID = "cnn"
ARCHITECTURE_NAME = "VGG-inspired CNN (explicit build_cnn_model)"
PRETRAINED_MODEL_ID = "microsoft/resnet-50"

# CNN Hyperparameters (used by build_cnn_model; ignored by frozen ResNet trunk)
NUM_KERNELS = 32  # filters in the first conv block; doubles each block
KERNEL_SIZE = (3, 3)
CONV_STRIDE = (1, 1)
POOL_STRIDE = (2, 2)
ACTIVATION_FUNC = "relu"
DROP = 0.2
SPATIAL_DROP = 0.2

# Extra MLP on the ResNet classifier head (0 disables; CNN uses its own Dense stack)
HEAD_MLP_HIDDEN = 512
DROPOUT_CLASSIFIER_HEAD = 0.3

# --- Class imbalance ---
# Keras-style class weights: multiply each sample's loss by its class weight.
CLASSES_UNBALANCED = True
# Oversample minority classes (WeightedRandomSampler). Approximate selective
# augmentation when we are not synthesizing extra images with ImageDataGenerator.
SEL_AUG = True

# --- Optimizer / schedule ---
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0
BATCH_SIZE = 16
DATALOADER_NUM_WORKERS = 0  # Windows: keep 0; Linux may use 4
EPOCHS = 25

# --- Backbone fine-tuning ---
# If True, freeze conv features for FREEZE_BACKBONE_EPOCHS then unfreeze.
FREEZE_BACKBONE = False
FREEZE_BACKBONE_EPOCHS = 3

# --- Splits (used by arth_tools.data.splits, unused if manifests already exist) ---
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
# Will attempt equal class mix if enough patients/samples
SPLIT_FORCE_EQ_DIST = False
# Splits on patient id to prevent images from the same patient across splits
SPLIT_FORCE_PT_STRAT = True

# --- Metrics / held-out eval ---
# 'accuracy' | 'f1' / 'macro_f1' | 'auc' / 'roc_auc'  (notebook used 'AUC')
PRIMARY_METRIC = "accuracy"
# After fit, score the test manifest with the best checkpoint (same z-score as train).
EVAL_ON_TEST = True
# Also report one-row-per-patient metrics (mean scores, then argmax / threshold).
EVAL_PATIENT_AGGREGATE = True
# Decision threshold for sigmoid / single-logit heads
BINARY_THRESHOLD = 0.5

# --- Callbacks ---
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

USE_REDUCE_LR = True
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN_LR = 1e-7

USE_MODEL_CHECKPOINT = True
CHECKPOINT_BEST_NAME = "best_model.pt"

# Freeze a copy under FROZEN_DIR / run_dir/frozen/ only if val metric meets this.
FREEZE_MIN_VALUE = 0.0
FREEZE_MIN_EPOCH = 1
FREEZE_MODE: Literal["max", "min"] = "max"

# --- Image preprocessing ---
RESIZE_DIMS = (224, 224)
INPUT_HEIGHT, INPUT_WIDTH = RESIZE_DIMS
INPUT_CHANNELS = 3
# Grayscale ultrasound → replicate to 3 channels (ResNet / 3-ch CNN)
REPLICATE_GRAYSCALE_TO_RGB = True
# Brightness jitter on the training loader (full geometric aug is still off-disk)
USE_TRAIN_JITTER = True
# Z-score using training-set pixel stats (IPFP used masked z-score; here the
# whole image is used because exported PNGs have no segmentation channel)
USE_TRAIN_ZSCORE = True

# --- Device / seed / logging ---
DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
SEED = 0
LOG_FILENAME_PREFIX = "train"
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL = "DEBUG"

# Per-run reports stay in the repo (curves, epochs.csv, freeze_decision).
REPORT_ROOT = TRAINING_REPORT_DIR

# Optional Optuna wrapper (off unless you flip this or pass HPOConfig)
HPO_ENABLED = False
HPO_N_TRIALS = 4
HPO_TIMEOUT_S: float | None = None
HPO_SEARCH: dict[str, Any] = {
    "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
    "batch_size": {"choices": [4, 8, 16]},
}


# ============================================================================
# Snapshot object (smoke / HPO / CLI overrides — not the thing you edit)
# ============================================================================


@dataclass
class HPOConfig:
    enabled: bool = False
    n_trials: int = 4
    timeout_s: float | None = None
    search: dict[str, Any] = field(default_factory=lambda: dict(HPO_SEARCH))


@dataclass
class TrainingConfig:
    """Mutable copy of the control board. Real runs: TrainingConfig() with no args."""

    architecture_id: str = ARCHITECTURE_ID
    architecture_name: str = ARCHITECTURE_NAME
    pretrained_id: str = PRETRAINED_MODEL_ID
    loss_name: str = LOSS_NAME
    output_type: str = OUTPUT_TYPE
    num_classes: int = NUM_CLASSES
    primary_metric: str = PRIMARY_METRIC

    num_kernels: int = NUM_KERNELS
    kernel_size: tuple[int, int] = KERNEL_SIZE
    conv_stride: tuple[int, int] = CONV_STRIDE
    pool_stride: tuple[int, int] = POOL_STRIDE
    activation_func: str = ACTIVATION_FUNC
    drop: float = DROP
    spatial_drop: float = SPATIAL_DROP
    head_mlp_hidden: int = HEAD_MLP_HIDDEN
    dropout_classifier_head: float = DROPOUT_CLASSIFIER_HEAD

    input_height: int = INPUT_HEIGHT
    input_width: int = INPUT_WIDTH
    input_channels: int = INPUT_CHANNELS
    replicate_grayscale_to_rgb: bool = REPLICATE_GRAYSCALE_TO_RGB
    train_jitter: bool = USE_TRAIN_JITTER
    train_zscore: bool = USE_TRAIN_ZSCORE

    report_root: Path = REPORT_ROOT
    run_id: str | None = None
    seed: int = SEED
    device: str = DEVICE

    train_manifest: Path | None = None
    val_manifest: Path | None = None
    test_manifest: Path | None = None
    checkpoint_dir: Path = CHECKPOINT_DIR
    frozen_dir: Path = FROZEN_DIR

    filepath_column: str = FILEPATH_COLUMN
    label_column: str = LABEL_COLUMN
    patient_id_column: str = PATIENT_ID_COLUMN

    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    dataloader_num_workers: int = DATALOADER_NUM_WORKERS

    freeze_backbone: bool = FREEZE_BACKBONE
    freeze_backbone_epochs: int = FREEZE_BACKBONE_EPOCHS
    class_weights: bool = CLASSES_UNBALANCED
    oversample_minority: bool = SEL_AUG

    use_early_stopping: bool = USE_EARLY_STOPPING
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA

    use_reduce_lr: bool = USE_REDUCE_LR
    reduce_lr_patience: int = REDUCE_LR_PATIENCE
    reduce_lr_factor: float = REDUCE_LR_FACTOR
    reduce_lr_min_lr: float = REDUCE_LR_MIN_LR

    use_model_checkpoint: bool = USE_MODEL_CHECKPOINT
    checkpoint_best_name: str = CHECKPOINT_BEST_NAME

    freeze_min_value: float = FREEZE_MIN_VALUE
    freeze_min_epoch: int = FREEZE_MIN_EPOCH
    freeze_mode: str = FREEZE_MODE

    train_split: float = TRAIN_SPLIT
    val_split: float = VAL_SPLIT
    test_split: float = TEST_SPLIT
    split_force_eq_dist: bool = SPLIT_FORCE_EQ_DIST
    split_force_pt_strat: bool = SPLIT_FORCE_PT_STRAT

    eval_on_test: bool = EVAL_ON_TEST
    eval_patient_aggregate: bool = EVAL_PATIENT_AGGREGATE
    binary_threshold: float = BINARY_THRESHOLD

    hpo: HPOConfig = field(
        default_factory=lambda: HPOConfig(
            enabled=HPO_ENABLED,
            n_trials=HPO_N_TRIALS,
            timeout_s=HPO_TIMEOUT_S,
            search=dict(HPO_SEARCH),
        )
    )

    def __post_init__(self) -> None:
        if self.train_manifest is None:
            self.train_manifest = MANIFEST_TRAIN
        if self.val_manifest is None:
            self.val_manifest = MANIFEST_VAL
        if self.test_manifest is None:
            self.test_manifest = MANIFEST_TEST
        self.train_manifest = Path(self.train_manifest) if self.train_manifest else None
        self.val_manifest = Path(self.val_manifest) if self.val_manifest else None
        self.test_manifest = Path(self.test_manifest) if self.test_manifest else None
        self.report_root = Path(self.report_root)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.frozen_dir = Path(self.frozen_dir)
        if isinstance(self.hpo, dict):
            self.hpo = HPOConfig(**self.hpo)
        self.kernel_size = tuple(self.kernel_size)  # type: ignore[assignment]
        self.conv_stride = tuple(self.conv_stride)  # type: ignore[assignment]
        self.pool_stride = tuple(self.pool_stride)  # type: ignore[assignment]

    def image_size(self) -> tuple[int, int]:
        return (int(self.input_width), int(self.input_height))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, val in list(data.items()):
            if isinstance(val, Path):
                data[key] = str(val)
        return data

    def hyperparameters_dict(self) -> dict[str, Any]:
        """Flat key: value map written to hyperparameters.txt (IPFP style)."""
        return {
            "loss_func": self.loss_name,
            "labels": self.label_column,
            "output": self.output_type,
            "architecture_id": self.architecture_id,
            "classes_unbalanced": self.class_weights,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "num_kernels": self.num_kernels,
            "kernel_size": self.kernel_size,
            "conv_stride": self.conv_stride,
            "pool_stride": self.pool_stride,
            "activation_func": self.activation_func,
            "metric": self.primary_metric,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "drop": self.drop,
            "spatial drop": self.spatial_drop,
            "Split with equal distributions": self.split_force_eq_dist,
            "Split on patient id": self.split_force_pt_strat,
            "Selective augmentation": self.oversample_minority,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "train_zscore": self.train_zscore,
            "train_jitter": self.train_jitter,
            "freeze_backbone": self.freeze_backbone,
            "seed": self.seed,
            "device": self.device,
            "eval_on_test": self.eval_on_test,
            "eval_patient_aggregate": self.eval_patient_aggregate,
            "binary_threshold": self.binary_threshold,
        }


def from_control_board(**overrides: Any) -> TrainingConfig:
    return TrainingConfig(**overrides)


def load_config_yaml(path: Path) -> TrainingConfig:
    """Optional YAML overlay on top of the control board (not required for training)."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raw = {}
    known = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in raw.items() if k in known}
    # Nested spec: from older YAML snapshots
    spec = raw.get("spec") if isinstance(raw.get("spec"), dict) else {}
    rename = {
        "architecture_id": "architecture_id",
        "architecture_name": "architecture_name",
        "pretrained_id": "pretrained_id",
        "loss_name": "loss_name",
        "output_type": "output_type",
        "num_classes": "num_classes",
        "primary_metric": "primary_metric",
        "input_height": "input_height",
        "input_width": "input_width",
        "input_channels": "input_channels",
        "replicate_grayscale_to_rgb": "replicate_grayscale_to_rgb",
    }
    for src, dest in rename.items():
        if dest not in filtered and src in spec:
            filtered[dest] = spec[src]
    freeze = spec.get("freeze") if isinstance(spec.get("freeze"), dict) else {}
    if "freeze_min_value" not in filtered and "min_value" in freeze:
        filtered["freeze_min_value"] = freeze["min_value"]
    if "freeze_min_epoch" not in filtered and "min_epoch" in freeze:
        filtered["freeze_min_epoch"] = freeze["min_epoch"]
    if "freeze_mode" not in filtered and "mode" in freeze:
        filtered["freeze_mode"] = freeze["mode"]
    return TrainingConfig(**filtered)


def save_hyperparameters_txt(cfg: TrainingConfig, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for key, value in cfg.hyperparameters_dict().items():
            fh.write(f"{key}: {value}\n")
    return dest


def new_run_id() -> str:
    return datetime.now().strftime("Fit_%Y%m%d-%H%M%S")


def checkpoint_path(cfg: TrainingConfig | None = None) -> Path:
    cfg = cfg or TrainingConfig()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return cfg.checkpoint_dir / cfg.checkpoint_best_name
