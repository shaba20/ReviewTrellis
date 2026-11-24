import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import os


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


print("\nReview Text:")

print(X.head())


print("\nReview Labels:")

print(y.head())


# --------------------------------
# STEP 3: CREATE TF-IDF VECTORIZER
# --------------------------------

tfidf = TfidfVectorizer(

    max_features=30000,

    ngram_range=(1, 2),

    min_df=2

)


# --------------------------------
# STEP 4: CONVERT TEXT INTO NUMBERS
# --------------------------------

X_tfidf = tfidf.fit_transform(X)


print("\nTF-IDF Conversion Completed")


# --------------------------------
# STEP 5: DISPLAY MATRIX INFORMATION
# --------------------------------

print("\nTF-IDF Matrix Shape:")

print(X_tfidf.shape)


print("\nNumber of Features:")

print(len(tfidf.get_feature_names_out()))


# --------------------------------
# STEP 6: DISPLAY FEATURE NAMES
# --------------------------------

feature_names = tfidf.get_feature_names_out()


print("\nFirst 20 TF-IDF Features:")

print(feature_names[:20])


# --------------------------------
# STEP 7: DISPLAY NUMERICAL VALUES
# --------------------------------

print("\nNumerical Representation of First Review:")

print(X_tfidf[0].toarray())


# --------------------------------
# STEP 8: SAVE FEATURE INFORMATION
# --------------------------------

os.makedirs(

    "reports",

    exist_ok=True

)


feature_df = pd.DataFrame({

    "Feature": feature_names

})


feature_df.to_csv(

    "reports/tfidf_features.csv",

    index=False

)


print("\nTF-IDF Feature Information Saved")

print("File: reports/tfidf_features.csv")


# --------------------------------
# FINAL MESSAGE
# --------------------------------

print("\nPART 7 COMPLETED SUCCESSFULLY")