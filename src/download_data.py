from datasets import load_dataset
import json

DATASETS = {
    "inconsistent": "Salesforce/FaithEval-inconsistent-v1.0",
    "unanswerable": "Salesforce/FaithEval-unanswerable-v1.0",
}

for name, hf_id in DATASETS.items():
    ds = load_dataset(hf_id)
    print(f"\n===== {name} =====")
    print(ds)

    for split_name, split in ds.items():
        out_path = f"data/raw/{name}_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for row in split:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(split)} rows -> {out_path}")

        first = split[0]
        print("\nFIRST ITEM:")
        for key, value in first.items():
            text = str(value)
            print(f"  {key}: {text[:600]}{'...' if len(text) > 600 else ''}")