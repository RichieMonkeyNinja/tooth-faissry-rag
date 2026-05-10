from datasets_rag import concatenate_datasets, load_dataset

ds = load_dataset("openlifescienceai/medmcqa")

combined = concatenate_datasets([ds["train"], ds["validation"]])
output_path = "medmcqa_train_validation.csv"
combined.to_csv(output_path, index=False)

print(ds)
print(ds["train"][0])
print(f"Saved {len(combined)} rows to {output_path}")
