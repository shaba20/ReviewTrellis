<div align="center">

# 🌳 ReviewTrellis

### Fake Product Review Detection & Product Trust Analysis

**Detect suspicious product reviews using Machine Learning and help users make more informed purchasing decisions.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-1F5F73)
![Best Model](https://img.shields.io/badge/Best%20Model-Linear%20SVM-success)
![Accuracy](https://img.shields.io/badge/Accuracy-94.55%25-brightgreen)

</div>

> ⚠️ **Disclaimer:** ReviewTrellis provides automated predictions based on patterns learned from its training dataset. A prediction is not definitive proof that a review is fake or genuine.

---

## 📑 Table of Contents

- [Website Preview](#-website-preview)
- [Project Overview](#-project-overview)
- [Project Objectives](#-project-objectives)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Dataset Information](#-dataset-information)
- [Data Preprocessing](#-data-preprocessing)
- [Algorithms Used](#-algorithms-used)
- [TF-IDF Feature Extraction](#-tf-idf-feature-extraction)
- [Model Performance](#-model-performance)
- [Graphs & Visualizations](#-graphs--visualizations)
- [Explainable Predictions](#-explainable-predictions)
- [Installation & Run Instructions](#️-installation--run-instructions)
- [Testing](#-testing)
- [Technologies Used](#️-technologies-used)
- [Limitations](#️-limitations)
- [Future Scope](#-future-scope)
- [Developed By](#-developed-by)

---

## 📸 Website Preview

![ReviewTrellis Website](screenshots/homepage.png)

> Save your website screenshot as `screenshots/homepage.png` so it appears here on GitHub.

---

## 📌 Project Overview

ReviewTrellis is a Machine Learning-based web application designed to identify potentially fake product reviews. Online shopping platforms contain a large number of customer reviews, but some reviews may be misleading, artificially generated, or written to manipulate product ratings.

ReviewTrellis analyzes the text of a review and predicts whether it is **Likely Genuine** or **Potentially Fake** using Natural Language Processing (NLP) and Machine Learning algorithms.

The project also provides a user-friendly website where users can enter a review and analyze its authenticity.

---

## 🎯 Project Objectives

- 🔎 Identify potentially fake product reviews.
- 📝 Apply Natural Language Processing techniques to review text.
- 🔤 Extract important textual features using TF-IDF.
- 🤖 Train and compare multiple Machine Learning algorithms.
- 📏 Evaluate model performance using suitable metrics.
- 🌐 Provide an interactive web application for review analysis.
- 🛒 Help users understand the reliability of online product reviews.

---

## 🔄 Machine Learning Workflow

```mermaid
flowchart TD
    A[Product Review Dataset] --> B[Data Preprocessing]
    B --> C[Exploratory Data Analysis]
    C --> D[TF-IDF Feature Extraction]
    D --> E[Train-Test Split]
    E --> F1[Logistic Regression]
    E --> F2[Multinomial Naive Bayes]
    E --> F3[Linear SVM]
    F1 --> G[Model Evaluation]
    F2 --> G
    F3 --> G
    G --> H[Best Model: Linear SVM]
    H --> I[Flask Backend & Prediction API]
    I --> J[ReviewTrellis Website]
    J --> K[Likely Genuine / Potentially Fake + Explanation]
```

The project follows a complete Machine Learning workflow:

1. **Dataset Collection** – Collect product review data with genuine and fake review labels.
2. **Data Preprocessing** – Clean the review text and prepare it for analysis.
3. **Exploratory Data Analysis (EDA)** – Analyze the dataset and understand review patterns.
4. **Feature Extraction** – Convert text into numerical features using TF-IDF.
5. **Model Training** – Train multiple Machine Learning algorithms.
6. **Model Evaluation** – Compare models using Accuracy, Precision, Recall, and F1 Score.
7. **Best Model Selection** – Select the best-performing model based on evaluation results.
8. **Web Application Integration** – Connect the trained model with the website.
9. **Review Prediction** – Allow users to submit a review and receive a prediction.
10. **Prediction Explanation** – Show review-specific TF-IDF feature contributions and detected writing-style cues alongside each result.

> The current Linear SVM does not provide calibrated class probabilities, so the site reports **Prediction confidence unavailable** instead of presenting its decision score as a confidence percentage. Explanations identify signals in the text and do not establish whether a review is authentic.

---

## 📂 Dataset Information

The project uses a product review dataset containing review text and corresponding labels.

### Dataset Files

| File | Description |
|---|---|
| `data/raw/Fake_Reviews.csv` | Original dataset |
| `data/processed/reviews_cleaned.csv` | Cleaned and processed dataset |

### Dataset Labels

| Label | Numeric Value | Meaning |
|---|:---:|---|
| OR | 0 | Genuine (original) review |
| CG | 1 | Computer-generated or fake review |

The dataset is used to train and evaluate the Machine Learning models.

**Dataset Source:** Add the original dataset download link here.

**Dataset License:** Verify the dataset's license before sharing the dataset publicly.

---

## 🧹 Data Preprocessing

The review text is prepared before feature extraction and model training.

Preprocessing may include:

- Handling missing values
- Removing duplicate records
- Cleaning unnecessary text
- Converting text into a consistent format
- Preparing review labels
- Splitting data into training and testing sets

The processed dataset is saved for further analysis and model training.

---

## 🧠 Algorithms Used

ReviewTrellis compares three Machine Learning algorithms.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm used to predict whether a review belongs to the genuine or fake category. It is used as the baseline model.

### 2. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification tasks.

### 3. Linear Support Vector Machine (Linear SVM)

Linear SVM is a supervised learning algorithm that identifies a decision boundary between different classes. It is effective for high-dimensional text data.

🏆 **Best Model:** Linear SVM was selected as the best-performing model based on the project's evaluation results.

---

## 🔤 TF-IDF Feature Extraction

ReviewTrellis uses **Term Frequency–Inverse Document Frequency (TF-IDF)** to convert review text into numerical features.

- **Term Frequency (TF):** how often a word appears in a review.
- **Inverse Document Frequency (IDF):** reduces the weight of words that appear in many reviews.
- **TF-IDF:** combines both to show how important a word is in a review relative to the whole collection of reviews.

It helps the Machine Learning models identify useful textual patterns for classification. The extracted TF-IDF features are used as input to the classification algorithms.

---

## 📊 Model Performance

The models are evaluated using the following performance metrics:

- **Accuracy:** Measures the overall proportion of correct predictions.
- **Precision:** Measures how many reviews predicted as fake are actually fake.
- **Recall:** Measures how many actual fake reviews are correctly identified.
- **F1 Score:** Combines Precision and Recall into a single metric.

### 📊 Performance Comparison

| Algorithm | Accuracy | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 92.98% | 94.51% | 91.25% | 92.85% |
| Multinomial Naive Bayes | 90.44% | 88.65% | 92.74% | 90.65% |
| **Linear SVM** 🏆 | **94.55%** | **94.86%** | **94.20%** | **94.53%** |

🏆 **Best Performing Model: Linear SVM** – it achieved the highest Accuracy, Precision, Recall, and F1 Score in this comparison, so it was selected as the final model.

---

## 📈 Graphs & Visualizations

The project includes visualizations to understand the dataset and compare model performance.

### 1️⃣ Accuracy Comparison

![Accuracy Comparison](reports/evaluation/accuracy_comparison.png)

**What this graph shows:** the test accuracy of the three algorithms, which is the percentage of all reviews (genuine and fake) that each model classified correctly.

**How to read it:**
- **Linear SVM – 94.55%:** the highest accuracy, so it makes the fewest mistakes overall.
- **Logistic Regression – 92.98%:** a strong baseline, about 1.6 percentage points behind Linear SVM.
- **Multinomial Naive Bayes – 90.44%:** the lowest of the three, because its simple word-independence assumption is less suited to this data.

### 2️⃣ Precision, Recall and F1 Score Comparison

![Precision Recall F1 Comparison](reports/evaluation/precision_recall_f1_comparison.png)

**What this graph shows:** how well each model performs on the fake-review class, using three metrics side by side.

**How to read it:**
- **Precision:** when the model says "fake", how often it is right. Linear SVM (94.86%) is slightly ahead of Logistic Regression (94.51%). Naive Bayes is lowest (88.65%), so it raises more false alarms.
- **Recall:** how many of the actual fake reviews the model catches. Linear SVM (94.20%) is highest. Naive Bayes (92.74%) beats Logistic Regression (91.25%) here, so it misses fewer fakes but at the cost of precision.
- **F1 Score:** the balance of precision and recall. Linear SVM (94.53%) leads, which is why it was chosen as the final model.

### 3️⃣ Exploratory Data Analysis

EDA visualizations are available in the `reports/eda/` directory.

These graphs help understand the dataset, review distribution, and patterns in the review text.

---

## 🔍 Explainable Predictions

Every prediction on the website includes a **"Why this prediction?"** section so users can see what the result was based on.

| Output | Description |
|---|---|
| **Prediction** | `Likely Genuine` or `Likely Fake` |
| **Confidence** | Shown only when the model provides a reliable probability. Otherwise: *Prediction confidence unavailable.* |
| **Reasons** | Two or three explanations built from the review text, including the TF-IDF terms that contributed most to the result |
| **Detected indicators** | Writing-style cues found in the text, such as promotional wording, first-person experience, product-use details, mixed opinions, recommendation phrases, repeated emphatic wording, or informal phrasing |

The prediction API returns structured JSON containing `prediction`, `confidence`, `reasons`, and `detected_indicators`.

> **Important:** these indicators describe linguistic patterns, not proof. A polished review is not automatically fake, and an unpolished review is not automatically genuine.

---

## ⚙️ Installation & Run Instructions

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd ReviewTrellis
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the Web Application

```bash
python -m app.server
```

### Step 7: Open the Website

Open your browser and visit:

```text
http://127.0.0.1:5000
```

The website will be available locally while the backend server is running.

To stop the server, press `Ctrl + C` in the terminal.

---

## 🧪 Testing

The project includes tests for checking the web application.

Run the tests using:

```bash
python -m unittest discover -s tests
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Backend web framework |
| HTML | Website structure |
| CSS | Website styling |
| JavaScript | Frontend interaction |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning algorithms |
| TF-IDF | Text feature extraction |
| Pickle | Saving the trained model |
| Jupyter Notebook | Data analysis and experimentation |
| Git & GitHub | Version control and project hosting |

---

## ⚠️ Limitations

- The model's performance depends on the quality and diversity of the training dataset.
- The model may learn dataset-specific writing patterns, such as review length or whether a text is complete.
- Some genuine reviews may be classified as potentially fake, and some fake reviews as likely genuine.
- Genuine reviews can also be promotional or emotional, and fake reviews can look natural.
- The model may not perform equally well on reviews from every product category.
- The system analyzes review text and may not verify the actual purchase or reviewer identity.
- The Linear SVM does not provide calibrated probabilities, so confidence is not shown as a percentage.
- Machine Learning predictions are estimates and should not be treated as absolute proof of review authenticity.

---

## 🚀 Future Scope

Possible improvements include:

- Adding more diverse product review datasets.
- Improving detection of AI-generated and manipulated reviews.
- Adding calibrated confidence estimates.
- Supporting multiple languages.
- Adding product-level trust analysis.
- Detecting suspicious reviewer behavior.
- Integrating deep learning and transformer-based NLP models.
- Deploying the application on a cloud platform.
- Improving the model using real-world user feedback.

---

## 👩‍💻 Developed By

**Shaba Khatoon**

🎓 B.Tech Computer Science & Engineering
🤖 Artificial Intelligence & Machine Learning

This project was developed as part of my Machine Learning learning journey to explore Natural Language Processing, text classification, model evaluation, explainable AI, and web application development.

---

## 🤝 Contributions

Suggestions, feedback, and contributions are welcome.

If you find an issue or have an idea to improve the project, feel free to open an issue or submit a pull request.

---

## ⭐ Support

If you find this project interesting or useful, please consider giving the repository a ⭐ on GitHub.

**Thank you for visiting ReviewTrellis!** 💙