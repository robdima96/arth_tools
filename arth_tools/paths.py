"""Repository and reporting paths (anchored to this package).

Data and model weights live on the removable disk; those paths are on the
CONTROL BOARD in arth_tools.training.config (DATA_ROOT, CHECKPOINT_DIR, ...).
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

PROMPT_HISTORY_DIR = REPO_ROOT / "prompt_history"
REPORTING_DIR = REPO_ROOT / "reporting"
TRAINING_REPORT_DIR = REPORTING_DIR / "training"
FIXTURES_DIR = REPO_ROOT / "fixtures"
