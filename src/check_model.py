import pandas as pd, glob
from predict import predict_review

path = glob.glob("data/processed/*.csv")[0]
df = pd.read_csv(path)
print("File:", path)
print("Columns:", df.columns.tolist())
print(df["label"].value_counts(), "\n")

text_col = "review_text" if "review_text" in df.columns else df.columns[0]

for lab in df["label"].unique():
    sample = df[df["label"] == lab].sample(30, random_state=1)
    preds = [predict_review(t) for t in sample[text_col]]
    print(f"True label {lab!r}: predicted 0 (genuine) = {preds.count(0)}, predicted 1 (fake) = {preds.count(1)}")
    print("   example:", sample[text_col].iloc[0][:200], "\n")