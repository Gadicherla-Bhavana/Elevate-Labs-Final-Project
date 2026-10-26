
from pathlib import Path
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


def find_dataset(filename):
    """Find a dataset CSV in the project folder or data folder."""
    possible_paths = [
        BASE_DIR / filename,
        DATA_DIR / filename,
    ]

    for path in possible_paths:
        if path.is_file():
            print(f"Found: {path}")
            return path

    raise FileNotFoundError(
        f"{filename} not found. Put it beside train_model.py "
        "or inside the data folder."
    )


def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("Loading the dataset...")

    fake = pd.read_csv(find_dataset("Fake.csv"))
    real = pd.read_csv(find_dataset("True.csv"))

    if "text" not in fake.columns or "text" not in real.columns:
        raise ValueError("Both CSV files must contain a 'text' column.")

    # Label: 0 = Fake, 1 = Real
    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real], ignore_index=True)
    data = data.dropna(subset=["text"]).copy()

    if "title" in data.columns:
        data["content"] = (
            data["title"].fillna("").astype(str)
            + " "
            + data["text"].astype(str)
        )
    else:
        data["content"] = data["text"].astype(str)

    data["content"] = data["content"].map(clean_text)
    data = data[data["content"].str.len() > 20]
    data = data.drop_duplicates(subset=["content"])

    print(f"Total articles: {len(data):,}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data["content"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    # Convert text into numerical features and train classifier
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
            ),
        ),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    print("Training the model. Please wait...")
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Fake", "Real"],
    )
    matrix = confusion_matrix(y_test, predictions)

    print(f"\nModel accuracy: {accuracy:.2%}")
    print("\nClassification report:\n")
    print(report)
    print("Confusion matrix:\n")
    print(matrix)

    # Save the trained model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "fake_news_model.joblib"
    joblib.dump(model, model_path)

    # Save evaluation results
    results = (
        f"Test accuracy: {accuracy:.2%}\n\n"
        + report
        + "\nConfusion matrix:\n"
        + str(matrix)
    )

    (BASE_DIR / "evaluation_results.txt").write_text(
        results,
        encoding="utf-8",
    )

    print(f"\nModel saved to: {model_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
