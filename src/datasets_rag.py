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
combined = combined.filter(lambda row: row["subject_name"] == "Dental")
combined = combined.map(
    lambda row: {
        "correct_ans": {
            0: row["opa"],
            1: row["opb"],
            2: row["opc"],
            3: row["opd"],
        }.get(row["cop"], "")
    }
)
combined = combined.select_columns(
    ["question", "opa", "opb", "opc", "opd", "cop", "correct_ans", "exp"]
)
output_path = project_root / "data/raw/medmcqa_data.csv"
combined.to_csv(str(output_path), index=False)

print(ds)
print(combined[0])
print(f"Saved {len(combined)} rows to {output_path}")
