import pandas as pd
import re
import os


# --------------------------------
# LOAD DATASET
# --------------------------------

df = pd.read_csv(
    "data/raw/Fake_Reviews.csv"
)


# --------------------------------
# SELECT COLUMNS
# --------------------------------

TEXT_COLUMN = "text_"

LABEL_COLUMN = "label"


df = df[
    [TEXT_COLUMN, LABEL_COLUMN]
]


# Rename columns

df.columns = [
    "review_text",
    "label"
]


# --------------------------------
# REMOVE MISSING VALUES
# --------------------------------

df = df.dropna()


# --------------------------------
# CLEAN TEXT
# --------------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["review_text"] = df[
    "review_text"
].apply(clean_text)


# --------------------------------
# REMOVE EMPTY REVIEWS
# --------------------------------

df = df[
    df["review_text"] != ""
]


# --------------------------------
# CONVERT LABELS
# --------------------------------

df["label"] = (
    df["label"]
    .astype(str)
    .str.upper()
    .str.strip()
)


df["label"] = df["label"].map({

    "OR": 0,

    "CG": 1

})


# Remove unknown labels

df = df.dropna(
    subset=["label"]
)


df["label"] = df[
    "label"
].astype(int)


# --------------------------------
# REMOVE DUPLICATES
# --------------------------------

df = df.drop_duplicates(
    subset=["review_text"]
)


# --------------------------------
# SAVE DATASET
# --------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)


df.to_csv(

    "data/processed/reviews_cleaned.csv",

    index=False

)


# --------------------------------
# FINAL INFORMATION
# --------------------------------

print("\nDataset Preparation Completed")

print("Total Reviews:", len(df))

print("\nLabel Distribution:")

print(df["label"].value_counts())

print("\nFinal Dataset Shape:")

print(df.shape)

print("\nSaved Successfully")