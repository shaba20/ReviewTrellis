import joblib
import math
import re
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
model = joblib.load(MODEL_PATH)

PROMOTIONAL_TERMS = (
    "amazing", "awesome", "best", "excellent", "fantastic", "flawless",
    "greatest", "incredible", "life-changing", "must-have", "perfect",
    "wonderful",
)
EXTREME_TERMS = PROMOTIONAL_TERMS + (
    "awful", "horrible", "terrible", "worst",
)
POSITIVE_TERMS = (
    "amazing", "awesome", "excellent", "good", "great", "love", "perfect",
    "wonderful",
)
NEGATIVE_TERMS = (
    "bad", "bulky", "disappointed", "flimsy", "hated", "poor", "terrible",
    "uncomfortable", "unfortunately",
)
RECOMMENDATION_TERMS = ("buy", "recommend", "must-have", "should get", "worth it")
USAGE_TERMS = (
    "bought", "installed", "used", "using", "wear", "wore", "tried",
    "tested", "cooked", "washed", "fits", "lasted",
)


def clean_review(text: str) -> str:
    """
    Apply the same text normalization used when preparing the training data.
    """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [
        term
        for term in terms
        if re.search(rf"\b{re.escape(term)}\b", text)
    ]


def _detect_indicators(review: str, cleaned_review: str) -> list[str]:
    indicators = []

    personal_terms = _find_terms(
        cleaned_review,
        ("i", "i'm", "i've", "my", "we", "we've", "our", "me"),
    )
    if personal_terms:
        indicators.append(
            f"First-person wording ({', '.join(personal_terms[:3])})"
        )

    usage_terms = _find_terms(cleaned_review, USAGE_TERMS)
    if usage_terms:
        indicators.append(
            f"Product-use language ({', '.join(usage_terms[:3])})"
        )

    specific_details = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|minutes?|"
        r"inches?|cm|mm|lbs?|kg|oz|ml|gb|tb)?\b",
        cleaned_review,
    )
    if specific_details:
        indicators.append(
            f"Specific quantities or measurements ({', '.join(specific_details[:2])})"
        )

    positive_terms = _find_terms(cleaned_review, POSITIVE_TERMS)
    negative_terms = _find_terms(cleaned_review, NEGATIVE_TERMS)
    if positive_terms and negative_terms:
        indicators.append(
            "Mixed opinion (positive and critical wording)"
        )

    promotional_terms = _find_terms(cleaned_review, PROMOTIONAL_TERMS)
    if promotional_terms:
        indicators.append(
            f"Promotional or emphatic praise ({', '.join(promotional_terms[:3])})"
        )

    recommendation_terms = _find_terms(cleaned_review, RECOMMENDATION_TERMS)
    if recommendation_terms:
        indicators.append(
            f"Recommendation language ({', '.join(recommendation_terms[:3])})"
        )

    extreme_terms = _find_terms(cleaned_review, EXTREME_TERMS)
    repeated_extremes = [
        term for term in extreme_terms
        if len(re.findall(rf"\b{re.escape(term)}\b", cleaned_review)) > 1
    ]
    if repeated_extremes or len(extreme_terms) > 1 or re.search(r"!{2,}", review):
        examples = repeated_extremes or extreme_terms
        evidence = f" ({', '.join(examples[:3])})" if examples else " (repeated exclamation marks)"
        indicators.append(f"Repeated or emphatic wording{evidence}")

    if promotional_terms and not personal_terms and not usage_terms:
        indicators.append(
            f"Broad praise without first-person or use details ({promotional_terms[0]})"
        )

    if re.search(r"\b(?:don't|isn't|it's|that's|can't|didn't|wasn't|won't)\b", cleaned_review):
        indicators.append("Informal phrasing (contraction present)")

    return indicators


def _top_model_terms(cleaned_review: str, prediction: int) -> list[str]:
    vectorizer = model.named_steps.get("tfidf")
    classifier = model.named_steps.get("classifier")
    coefficients = getattr(classifier, "coef_", None)
    if vectorizer is None or coefficients is None or len(classifier.classes_) != 2:
        return []

    features = vectorizer.transform([cleaned_review])
    contributions = features.multiply(coefficients[0]).toarray()[0]
    classes = list(classifier.classes_)
    direction = 1 if prediction == classes[1] else -1
    feature_names = vectorizer.get_feature_names_out()
    ranked_indexes = sorted(
        range(len(contributions)),
        key=lambda index: contributions[index] * direction,
        reverse=True,
    )
    return [
        str(feature_names[index])
        for index in ranked_indexes
        if contributions[index] * direction > 0
    ][:3]


def explain_review(review: str) -> dict:
    """Return a model prediction and review-specific explanation signals."""
    cleaned_review = clean_review(review)
    if not cleaned_review:
        raise ValueError("Please enter valid review text.")

    predicted_class = int(model.predict([cleaned_review])[0])
    is_genuine = predicted_class == 0
    prediction = "Likely Genuine" if is_genuine else "Likely Fake"
    model_terms = _top_model_terms(cleaned_review, predicted_class)
    indicators = _detect_indicators(review, cleaned_review)

    if model_terms:
        reasons = [
            f"TF-IDF terms {', '.join(f'“{term}”' for term in model_terms)} "
            f"had the strongest contributions toward the {prediction} result."
        ]
    else:
        reasons = [
            "No in-vocabulary TF-IDF terms contributed directly to this result."
        ]

    indicator_reasons = {
        "First-person wording": (
            "First-person wording appears in the review; this is a personal-style "
            "cue, not proof of a purchase."
        ),
        "Product-use language": (
            "The review describes product use, a concrete experience cue that does "
            "not establish authenticity by itself."
        ),
        "Specific quantities or measurements": (
            "The review includes a measurable detail; that specificity is a writing "
            "signal, not independent verification."
        ),
        "Mixed opinion": (
            "Both positive and critical language appear, giving the review a mixed tone."
        ),
        "Promotional or emphatic praise": (
            "Promotional or emphatic praise appears; this describes wording, not "
            "whether the review is genuine."
        ),
        "Recommendation language": (
            "A recommendation phrase appears; this is a stylistic cue rather than "
            "proof of intent."
        ),
        "Repeated or emphatic wording": (
            "Extreme wording or punctuation is repeated, creating an emphatic tone."
        ),
        "Broad praise without first-person or use details": (
            "The praise is broad and is not accompanied by first-person or product-use "
            "details in this text."
        ),
        "Informal phrasing": (
            "A contraction gives the review an informal writing-style cue."
        ),
    }
    for indicator in indicators:
        for category, reason in indicator_reasons.items():
            if indicator.startswith(category):
                reasons.append(reason)
                break
        if len(reasons) == 3:
            break

    if len(reasons) < 3 and not indicators:
        reasons.append(
            f"The text contains {len(cleaned_review.split())} words, and no strong "
            "writing-style cue from the current checks was detected."
        )

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_review])[0]
        class_index = list(model.classes_).index(predicted_class)
        candidate = float(probabilities[class_index])
        if math.isfinite(candidate) and 0 <= candidate <= 1:
            confidence = candidate

    return {
        "prediction": prediction,
        "predicted_label": "genuine" if is_genuine else "fake",
        "confidence": confidence,
        "reasons": reasons,
        "detected_indicators": indicators,
    }


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