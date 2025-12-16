import pandas as pd
import json

CURRENT_DATA_PATH = "current_data.csv"
OUTPUT_JSON_PATH = "predict_batch_payload.json"

THRESHOLD = 0.5

df = pd.read_csv(CURRENT_DATA_PATH)

drop_cols = ["target", "bad_proba", "bad_pred", "timestamp"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

records = df.to_dict(orient="records")

payload = {
    "records": records,
    "threshold": THRESHOLD
}


with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"Saved JSON with {len(records)}")
