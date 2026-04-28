from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
SAMPLES_DIR = PROJECT_ROOT / "samples"


def ensure_project_dirs() -> None:
    """Create local data/artifact folders used by scripts and notebooks."""
    for path in (DATA_DIR, FEATURES_DIR, CHECKPOINTS_DIR, REPORTS_DIR, SAMPLES_DIR):
        path.mkdir(parents=True, exist_ok=True)

