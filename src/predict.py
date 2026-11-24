import joblib
import re
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
model = joblib.load(MODEL_PATH)


def clean_review(text: str) -> str:
    """
    Apply the same text normalization used when preparing the training data.
    """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_review(review: str) -> int:
    """
    Return 0 for likely genuine and 1 for potentially fake.
    """
    cleaned_review = clean_review(review)
    if cleaned_review == "":
        raise ValueError("Please enter valid review text.")
    prediction = model.predict([cleaned_review])[0]
    return int(prediction)


if __name__ == "__main__":
    review = input("Enter a product review: ")
    try:
        prediction = predict_review(review)
    except ValueError as error:
        print(error)
    else:
        result = "Likely Genuine Review" if prediction == 0 else "Potentially Fake Review"
        print(f"\nPrediction: {result}")