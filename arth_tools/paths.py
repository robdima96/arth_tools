"""Repository and reporting paths (anchored to this package).

DICOM input and derived artifacts (DATA_ROOT, DICOM_ROOT) are on the
CONTROL BOARD in arth_tools.training.config.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

REPORTING_DIR = REPO_ROOT / "reporting"
TRAINING_REPORT_DIR = REPORTING_DIR / "training"
FIXTURES_DIR = REPO_ROOT / "fixtures"
CONFIGS_DIR = REPO_ROOT / "configs"
