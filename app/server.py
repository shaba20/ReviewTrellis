import csv
import math
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.predict import clean_review, explain_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_REVIEW_LENGTH = 10_000
EVALUATION_CSV = PROJECT_ROOT / "reports" / "evaluation" / "evaluation_summary.csv"
EVALUATION_METRICS = (
    {"key": "Accuracy", "label": "Test Accuracy"},
    {"key": "Precision", "label": "Precision"},
    {"key": "Recall", "label": "Recall"},
    {"key": "F1 Score", "label": "F1 Score"},
)
MODEL_STYLES = {
    "Linear SVM": {
        "label_class": "text-petrol font-semibold",
        "bar_class": "bg-petrol",
        "value_class": "font-bold text-ink",
    },
    "Logistic Regression": {
        "label_class": "text-muted",
        "bar_class": "bg-amber",
        "value_class": "text-muted",
    },
    "Multinomial Naive Bayes": {
        "label_class": "text-muted",
        "bar_class": "bg-[#9A9386]",
        "value_class": "text-muted",
    },
}

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.get("/")
def index():
    return render_template(
        "index.html",
        evaluation_metrics=EVALUATION_METRICS,
        evaluation_models=load_evaluation_models(),
    )


@app.after_request
def prevent_homepage_caching(response):
    if request.endpoint == "index":
        response.headers["Cache-Control"] = "no-store"
    return response


def load_evaluation_models():
    metric_keys = {metric["key"] for metric in EVALUATION_METRICS}
    required_columns = {"Algorithm", *metric_keys}
    results = {}

    with EVALUATION_CSV.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{EVALUATION_CSV} must contain columns: "
                f"{', '.join(sorted(required_columns))}."
            )

        for row_number, row in enumerate(reader, start=2):
            algorithm = (row.get("Algorithm") or "").strip()
            if algorithm not in MODEL_STYLES:
                raise ValueError(
                    f"{EVALUATION_CSV} has an unexpected algorithm on row {row_number}."
                )
            if algorithm in results:
                raise ValueError(
                    f"{EVALUATION_CSV} contains duplicate results for {algorithm}."
                )

            metrics = {}
            for metric_key in metric_keys:
                try:
                    value = float(row[metric_key])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{EVALUATION_CSV} has an invalid {metric_key} for {algorithm}."
                    ) from error
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(
                        f"{EVALUATION_CSV} has an out-of-range {metric_key} for {algorithm}."
                    )
                metrics[metric_key] = value
            results[algorithm] = metrics

    missing_algorithms = MODEL_STYLES.keys() - results.keys()
    if missing_algorithms:
        raise ValueError(
            f"{EVALUATION_CSV} is missing results for: "
            f"{', '.join(sorted(missing_algorithms))}."
        )

    return [
        {
            "algorithm": algorithm,
            "metrics": results[algorithm],
            **MODEL_STYLES[algorithm],
        }
        for algorithm in MODEL_STYLES
    ]


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    if not request.is_json:
        return api_error_response("Send the review as a JSON request.", 415)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return api_error_response("The request body must be a JSON object.", 400)

    review = payload.get("review")
    if not isinstance(review, str):
        return api_error_response("The 'review' field must be a string.", 400)

    review = review.strip()
    if not review:
        return api_error_response("Please enter a product review.", 400)
    if len(review) > MAX_REVIEW_LENGTH:
        return api_error_response(
            f"Reviews must be {MAX_REVIEW_LENGTH:,} characters or fewer.",
            413,
        )
    if not clean_review(review):
        return api_error_response("Please enter valid review text.", 400)

    analysis = explain_review(review)
    predicted_label = analysis["predicted_label"]
    message = analysis["prediction"]
    return jsonify({
        "success": True,
        "prediction": analysis["prediction"],
        "confidence": analysis["confidence"],
        "reasons": analysis["reasons"],
        "detected_indicators": analysis["detected_indicators"],
        "predicted_label": predicted_label,
        "message": message,
        "label": predicted_label,
        "result": message,
        "characters": len(review),
        "words": len(review.split()),
        "model": "Linear SVM",
    })


def api_error_response(message, status_code, **extra_fields):
    payload = {
        "success": False,
        "message": message,
        "error": message,
    }
    payload.update(extra_fields)
    return jsonify(payload), status_code


@app.errorhandler(413)
def request_too_large(_error):
    return api_error_response("The request is too large.", 413)


@app.errorhandler(HTTPException)
def handle_http_exception(error):
    if request.path.startswith("/api/"):
        extra_fields = {}
        if error.code == 405 and error.valid_methods:
            extra_fields["allowed_methods"] = error.valid_methods
        return api_error_response(
            error.description,
            error.code or 500,
            **extra_fields,
        )
    return error


@app.errorhandler(500)
def handle_internal_error(error):
    if request.path.startswith("/api/"):
        return api_error_response(
            "The server could not analyze this review. Check the server logs.",
            500,
        )
    return error


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
