import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------
# STEP 1: LOAD CLEANED DATASET
# --------------------------------

df = pd.read_csv(
    "data/processed/reviews_cleaned.csv"
)

print("Dataset Loaded Successfully")

print("Total Reviews:", len(df))


# --------------------------------
# STEP 2: SEPARATE TEXT AND LABEL
# --------------------------------

X = df["review_text"]

y = df["label"]


print("\nReview Text and Labels Separated")


# --------------------------------
# STEP 3: TRAIN-TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining Reviews:", len(X_train))

print("Testing Reviews:", len(X_test))


# --------------------------------
# STEP 4: DEFINE THREE ALGORITHMS
# --------------------------------

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=1000
        ),


    "Multinomial Naive Bayes":

        MultinomialNB(),


    "Linear SVM":

        LinearSVC()

}


# --------------------------------
# STEP 5: CREATE RESULTS STORAGE
# --------------------------------

results = []

best_model = None

best_f1 = -1

best_model_name = ""


# --------------------------------
# STEP 6: TRAIN EACH ALGORITHM
# --------------------------------

for name, algorithm in models.items():

    print("\n================================")

    print("Training Model:", name)

    print("================================")


    # Create ML pipeline

    model = Pipeline([

        (

            "tfidf",

            TfidfVectorizer(

                max_features=30000,

                ngram_range=(1, 2),

                min_df=2

            )

        ),

        (

            "classifier",

            algorithm

        )

    ])


    # Train model

    model.fit(

        X_train,

        y_train

    )


    print("Training Completed")


    # --------------------------------
    # STEP 7: MAKE PREDICTIONS
    # --------------------------------

    predictions = model.predict(

        X_test

    )


    # --------------------------------
    # STEP 8: CALCULATE METRICS
    # --------------------------------

    accuracy = accuracy_score(

        y_test,

        predictions

    )


    precision = precision_score(

        y_test,

        predictions,

        zero_division=0

    )


    recall = recall_score(

        y_test,

        predictions,

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        predictions,

        zero_division=0

    )


    # --------------------------------
    # STEP 9: STORE RESULTS
    # --------------------------------

    results.append({

        "Algorithm": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1

    })


    # --------------------------------
    # STEP 10: DISPLAY RESULTS
    # --------------------------------

    print("\nAccuracy:", accuracy)

    print("Precision:", precision)

    print("Recall:", recall)

    print("F1 Score:", f1)


    print("\nClassification Report:")

    print(

        classification_report(

            y_test,

            predictions,

            zero_division=0

        )

    )


    print("\nConfusion Matrix:")

    print(

        confusion_matrix(

            y_test,

            predictions

        )

    )


    # --------------------------------
    # STEP 11: SELECT BEST MODEL
    # --------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_model_name = name


# --------------------------------
# STEP 12: CREATE COMPARISON TABLE
# --------------------------------

results_df = pd.DataFrame(

    results

)


print("\n================================")

print("MODEL COMPARISON")

print("================================")


print(

    results_df.to_string(

        index=False

    )

)


# --------------------------------
# STEP 13: SAVE COMPARISON RESULTS
# --------------------------------

os.makedirs(

    "reports",

    exist_ok=True

)


results_df.to_csv(

    "reports/model_comparison.csv",

    index=False

)


print("\nModel Comparison Saved")

print("File: reports/model_comparison.csv")


# --------------------------------
# STEP 14: SAVE BEST MODEL
# --------------------------------

os.makedirs(

    "models",

    exist_ok=True

)


joblib.dump(

    best_model,

    "models/best_model.pkl"

)


print("\n================================")

print("BEST MODEL")

print("================================")


print("Best Algorithm:", best_model_name)

print("Best F1 Score:", best_f1)


print("\nBest Model Saved Successfully")

print("File: models/best_model.pkl")


# --------------------------------
# FINAL MESSAGE
# --------------------------------

print("\nPART 8 COMPLETED SUCCESSFULLY")