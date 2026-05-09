from pathlib import Path
import sys

# Avoid importing this file instead of the Hugging Face `datasets` package.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path = [
    path for path in sys.path if Path(path or ".").resolve() not in {current_dir, project_root}
]

from datasets import concatenate_datasets, load_dataset


ds = load_dataset("openlifescienceai/medmcqa")

combined = concatenate_datasets([ds["train"], ds["validation"]])
combined = combined.select_columns(["question", "exp"])
output_path = project_root / "data/raw/medmcqa_data.csv"
combined.to_csv(str(output_path), index=False)

print(ds)
print(combined[0])
print(f"Saved {len(combined)} rows to {output_path}")
