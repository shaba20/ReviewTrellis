import csv
import re
import unittest
from unittest.mock import patch

from app.server import EVALUATION_CSV, EVALUATION_METRICS, app


class ReviewApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_homepage_serves_frontend(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers["Cache-Control"])
        self.assertIn("/api/predict", html)
        self.assertRegex(
            html,
            r"fetch\('/api/predict',\s*\{\s*method:\s*'POST'",
        )
        self.assertNotIn("Sample Presets:", html)
        self.assertIn('id="review-textarea"', html)
        self.assertIn("Why this prediction?", html)
        self.assertIn('id="result-confidence"', html)
        self.assertIn('id="result-reasons"', html)
        self.assertIn('id="result-indicators"', html)

    def test_live_server_template_redirects_to_flask(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertIn('window.location.port === "5500"', html)
        self.assertIn('flaskUrl.port = "5000"', html)
        self.assertIn("window.location.replace(flaskUrl.href)", html)

    def test_homepage_metric_chart_matches_evaluation_summary_csv(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertNotIn("Sample values — replace with actual results", html)
        self.assertNotRegex(html, r"{%|{{")
        self.assertEqual(html.count("Comparative Model Metrics"), 1)

        with EVALUATION_CSV.open("r", encoding="utf-8-sig", newline="") as csv_file:
            models = list(csv.DictReader(csv_file))
            rendered_pairs = re.findall(
                r'data-algorithm="([^"]+)"\s+data-metric="([^"]+)"',
                html,
            )
            expected_pairs = {
                (model["Algorithm"], metric["key"])
                for model in models
                for metric in EVALUATION_METRICS
            }
            self.assertEqual(set(rendered_pairs), expected_pairs)
            self.assertEqual(len(rendered_pairs), len(expected_pairs))

            for model in models:
                for metric in EVALUATION_METRICS:
                    metric_key = metric["key"]
                    percentage = f"{float(model[metric_key]) * 100:.2f}"
                    row_pattern = re.compile(
                        rf'<div\s+class="flex items-center gap-3"\s+'
                        rf'data-algorithm="{re.escape(model["Algorithm"])}"\s+'
                        rf'data-metric="{re.escape(metric_key)}"\s+'
                        rf'data-value="{percentage}"\s*>.*?'
                        rf'<span[^>]*>{percentage}%</span>',
                        re.DOTALL,
                    )
                    with self.subTest(
                        algorithm=model["Algorithm"],
                        metric=metric_key,
                    ):
                        self.assertRegex(html, row_pattern)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_prediction_route_requires_post_and_reports_allowed_method(self):
        response = self.client.get("/api/predict")

        self.assertEqual(response.status_code, 405)
        self.assertIn("application/json", response.headers["Content-Type"])
        self.assertIn("POST", response.get_json()["allowed_methods"])

    def test_predict_returns_model_result_and_statistics(self):
        review = "I use this product every day and it works well."
        response = self.client.post("/api/predict", json={"review": review})

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers["Content-Type"])
        result = response.get_json()
        self.assertTrue(result["success"])
        self.assertIn(result["prediction"], {"Likely Genuine", "Likely Fake"})
        self.assertIsNone(result["confidence"])
        self.assertGreaterEqual(len(result["reasons"]), 2)
        self.assertLessEqual(len(result["reasons"]), 3)
        self.assertIsInstance(result["detected_indicators"], list)
        self.assertIn(result["predicted_label"], {"genuine", "fake"})
        expected_result = (
            "Likely Genuine"
            if result["predicted_label"] == "genuine"
            else "Likely Fake"
        )
        self.assertEqual(result["message"], expected_result)
        self.assertEqual(result["characters"], len(review))
        self.assertEqual(result["words"], len(review.split()))
        self.assertEqual(result["model"], "Linear SVM")

    def test_promotional_review_returns_frontend_prediction_contract(self):
        review = (
            "This is an absolutely amazing product! The outstanding quality and "
            "exceptional performance make it the perfect choice for everyone. "
            "Highly recommended!"
        )
        response = self.client.post("/api/predict", json={"review": review})

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertTrue(result["success"])
        self.assertIn(result["prediction"], {"Likely Genuine", "Likely Fake"})
        self.assertIsNone(result["confidence"])
        self.assertIsInstance(result["reasons"], list)
        self.assertGreaterEqual(len(result["reasons"]), 2)
        self.assertTrue(all(isinstance(reason, str) for reason in result["reasons"]))
        self.assertIsInstance(result["detected_indicators"], list)
        self.assertTrue(
            all(isinstance(indicator, str) for indicator in result["detected_indicators"])
        )

    def test_explanations_cover_varied_review_styles_without_forcing_labels(self):
        reviews = (
            "Buy this amazing product now! Best ever. Highly recommend!",
            (
                "This incredible, perfect product is absolutely amazing and the best "
                "thing ever. Buy it now, you will love it! I highly recommend it to "
                "everyone because it is fantastic, flawless, and wonderful."
            ),
            "I use this kettle daily; it heats water quickly.",
            (
                "I bought this kettle three weeks ago and use it every morning. It "
                "heats water quickly, although the lid feels flimsy and the handle "
                "gets warm. It is useful for tea, but I wish it held more water."
            ),
            (
                "I have used these headphones for two months. The sound is great, "
                "but the fit is uncomfortable after an hour. Battery life lasts "
                "about 8 hours, which works for my commute, though the case is bulky."
            ),
        )
        results = []

        for review in reviews:
            with self.subTest(review=review[:40]):
                response = self.client.post("/api/predict", json={"review": review})
                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                self.assertIn(result["prediction"], {"Likely Genuine", "Likely Fake"})
                self.assertIsNone(result["confidence"])
                self.assertGreaterEqual(len(result["reasons"]), 2)
                self.assertLessEqual(len(result["reasons"]), 3)
                self.assertIsInstance(result["detected_indicators"], list)
                results.append(result)

        self.assertTrue(
            any("Promotional or emphatic praise" in indicator
                for indicator in results[0]["detected_indicators"])
        )
        self.assertTrue(
            any("First-person wording" in indicator
                for indicator in results[2]["detected_indicators"])
        )
        self.assertTrue(
            any("Mixed opinion" in indicator
                for result in results[3:]
                for indicator in result["detected_indicators"])
        )
        self.assertGreater(
            len({tuple(result["reasons"]) for result in results}),
            1,
        )

    def test_predict_rejects_empty_or_non_text_reviews(self):
        for payload in ({"review": "  "}, {"review": "<b></b>"}, {"review": 42}, {}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/predict", json=payload)
                self.assertEqual(response.status_code, 400)
                result = response.get_json()
                self.assertFalse(result["success"])
                self.assertEqual(result["message"], result["error"])

    def test_predict_rejects_malformed_json_with_json_error(self):
        response = self.client.post(
            "/api/predict",
            data="{invalid json",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("application/json", response.headers["Content-Type"])
        self.assertIn("error", response.get_json())

    def test_predict_rejects_non_json_requests(self):
        response = self.client.post("/api/predict", data="review text")

        self.assertEqual(response.status_code, 415)
        self.assertIn("error", response.get_json())

    def test_predict_rejects_reviews_over_limit(self):
        response = self.client.post(
            "/api/predict",
            json={"review": "a" * 10_001},
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())

    def test_api_http_errors_are_json(self):
        response = self.client.get("/api/not-found")

        self.assertEqual(response.status_code, 404)
        self.assertIn("application/json", response.headers["Content-Type"])
        self.assertIn("error", response.get_json())
        self.assertFalse(response.get_json()["success"])

    def test_prediction_failure_returns_json_error(self):
        app.config["PROPAGATE_EXCEPTIONS"] = False
        try:
            with patch("src.predict.model.predict", side_effect=RuntimeError("test error")):
                response = self.client.post(
                    "/api/predict",
                    json={"review": "A valid product review."},
                )
        finally:
            app.config["PROPAGATE_EXCEPTIONS"] = None

        self.assertEqual(response.status_code, 500)
        self.assertIn("application/json", response.headers["Content-Type"])
        self.assertIn("error", response.get_json())
        self.assertFalse(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
