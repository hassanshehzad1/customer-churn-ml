"""Script to verify baseline model definitions."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.train import get_baseline_models


if __name__ == "__main__":
    models = get_baseline_models()

    print("Baseline Models Dictionary:")
    print("=" * 50)
    for name, model in models.items():
        print(f"Model Name: {name}")
        print(f"Model Type: {type(model)}")
        print("-" * 50)
