from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "03_news_labeler"

ARTICLES_FILE = ROOT / "outputs" / "02_news_data" / "news_articles.csv"
MANUAL_LABELS_FILE = DATA_DIR / "manual_labels.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_text_model() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def load_training_data() -> pd.DataFrame:
    articles = pd.read_csv(ARTICLES_FILE, dtype={"article_id": "string"})
    manual = pd.read_csv(MANUAL_LABELS_FILE, dtype={"article_id": "string"})
    manual = manual.dropna(subset=["manual_sentiment", "manual_category"]).copy()
    manual["manual_sentiment"] = manual["manual_sentiment"].astype(str).str.strip().str.lower()
    manual["manual_category"] = manual["manual_category"].astype(str).str.strip().str.lower()

    data = articles.merge(manual, on="article_id", how="inner")
    data["text"] = data["title_clean"].fillna(data["title"]).fillna("").astype(str)
    data = data[data["text"].str.len().gt(0)].copy()

    if data.empty:
        raise ValueError("No labeled rows found after joining manual_labels.csv with news_articles.csv")
    return data.reset_index(drop=True)


def evaluate_model(name: str, model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> tuple[dict, pd.DataFrame]:
    pred = model.predict(x_test)
    labels = sorted(y_test.unique().tolist())
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "task": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_test)),
    }
    for label in labels:
        metrics[f"f1_{label}"] = float(report[label]["f1-score"])

    matrix = confusion_matrix(y_test, pred, labels=labels)
    confusion = pd.DataFrame(matrix, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    return metrics, confusion


def predicted_frame(articles: pd.DataFrame, manual: pd.DataFrame, category_model: Pipeline, sentiment_model: Pipeline) -> pd.DataFrame:
    output = articles.copy()
    output["text"] = output["title_clean"].fillna(output["title"]).fillna("").astype(str)

    category_pred = category_model.predict(output["text"])
    sentiment_pred = sentiment_model.predict(output["text"])
    category_proba = category_model.predict_proba(output["text"])
    sentiment_proba = sentiment_model.predict_proba(output["text"])

    output["model_category"] = category_pred
    output["model_category_confidence"] = category_proba.max(axis=1).round(6)
    output["model_sentiment"] = sentiment_pred
    output["model_sentiment_confidence"] = sentiment_proba.max(axis=1).round(6)

    manual_clean = manual.dropna(subset=["manual_sentiment", "manual_category"]).copy()
    manual_clean["manual_sentiment"] = manual_clean["manual_sentiment"].astype(str).str.strip().str.lower()
    manual_clean["manual_category"] = manual_clean["manual_category"].astype(str).str.strip().str.lower()
    output = output.merge(manual_clean, on="article_id", how="left")

    has_manual = output["manual_category"].notna() & output["manual_sentiment"].notna()
    output["final_category_ml"] = output["model_category"]
    output["final_sentiment_ml"] = output["model_sentiment"]
    output.loc[has_manual, "final_category_ml"] = output.loc[has_manual, "manual_category"]
    output.loc[has_manual, "final_sentiment_ml"] = output.loc[has_manual, "manual_sentiment"]
    output["label_source_ml"] = "model"
    output.loc[has_manual, "label_source_ml"] = "manual"

    sentiment_score = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    output["final_general_sentiment_ml"] = output["final_sentiment_ml"].map(sentiment_score).fillna(0.0)

    return output[
        [
            "article_id",
            "published_date",
            "event_trading_date",
            "title",
            "category",
            "sentiment_label",
            "general_sentiment",
            "manual_category",
            "manual_sentiment",
            "model_category",
            "model_category_confidence",
            "model_sentiment",
            "model_sentiment_confidence",
            "final_category_ml",
            "final_sentiment_ml",
            "final_general_sentiment_ml",
            "label_source_ml",
        ]
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_training_data()
    articles = pd.read_csv(ARTICLES_FILE, dtype={"article_id": "string"})
    manual = pd.read_csv(MANUAL_LABELS_FILE, dtype={"article_id": "string"})

    category_train, category_test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["manual_category"],
    )
    sentiment_train, sentiment_test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["manual_sentiment"],
    )

    category_eval_model = make_text_model()
    sentiment_eval_model = make_text_model()

    category_eval_model.fit(category_train["text"], category_train["manual_category"])
    sentiment_eval_model.fit(sentiment_train["text"], sentiment_train["manual_sentiment"])

    category_metrics, category_confusion = evaluate_model(
        "category", category_eval_model, category_test["text"], category_test["manual_category"]
    )
    sentiment_metrics, sentiment_confusion = evaluate_model(
        "sentiment", sentiment_eval_model, sentiment_test["text"], sentiment_test["manual_sentiment"]
    )

    category_model = make_text_model()
    sentiment_model = make_text_model()
    category_model.fit(data["text"], data["manual_category"])
    sentiment_model.fit(data["text"], data["manual_sentiment"])

    metrics = pd.DataFrame([category_metrics, sentiment_metrics])
    predictions = predicted_frame(articles, manual, category_model, sentiment_model)

    metrics.to_csv(OUT_DIR / "news_labeler_metrics.csv", index=False)
    category_confusion.to_csv(OUT_DIR / "category_confusion_matrix.csv", encoding="utf-8-sig")
    sentiment_confusion.to_csv(OUT_DIR / "sentiment_confusion_matrix.csv", encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "news_articles_ml_labels.csv", index=False, encoding="utf-8-sig")
    joblib.dump(category_model, OUT_DIR / "category_labeler.joblib")
    joblib.dump(sentiment_model, OUT_DIR / "sentiment_labeler.joblib")

    summary = {
        "manual_labeled_rows": int(len(data)),
        "category_train_rows": int(len(category_train)),
        "category_test_rows": int(len(category_test)),
        "sentiment_train_rows": int(len(sentiment_train)),
        "sentiment_test_rows": int(len(sentiment_test)),
        "all_articles_labeled": int(len(predictions)),
        "manual_labels_applied": int(predictions["label_source_ml"].eq("manual").sum()),
        "model_labels_applied": int(predictions["label_source_ml"].eq("model").sum()),
    }
    with open(OUT_DIR / "news_labeler_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Generated ML news labels.")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
