import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "wine-categorizer.joblib")
CSV_PATH = os.path.join(BASE_DIR, "wine.csv")


def main():
    df = pd.read_csv(CSV_PATH, sep=",")

    if "cultivar" not in df.columns:
        raise ValueError("Expected a 'cultivar' column in wine.csv (target label).")

    # ---- FEATURES: MUST match your CSV column names exactly ----
    FEATURES = [
        "alcohol",
        "total phenols",
        "flavanoids",
        "proanthocyanins",
        "color intensity",
    ]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\nColumns: {list(df.columns)}")

    X = df[FEATURES].copy()

    enc = LabelEncoder()
    y = enc.fit_transform(df["cultivar"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    grid = GridSearchCV(
        pipe,
        param_grid={"knn__n_neighbors": range(1, 50)},
        cv=20,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("Best params:", grid.best_params_)
    print("Test accuracy:", round(acc, 4))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save everything Flask needs
    bundle = {
        "model": best_model,
        "feature_names": FEATURES,
        "classes": enc.classes_.tolist(),
    }
    joblib.dump(bundle, MODEL_PATH)
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
