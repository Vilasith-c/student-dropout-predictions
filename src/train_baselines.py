import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import XGBClassifier  # pip install xgboost


FIG_DIR = r"student-dropout-predictions\reports\figures"
os.makedirs(FIG_DIR, exist_ok=True)

def plot_feature_importance(model, feature_names, title, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # top 15

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)),
             importances[indices][::-1],
             align="center")
    plt.yticks(range(len(indices)),
               [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.close()

def train_logistic_regression(X_train, X_test, y_train_enc, y_test_enc, class_names):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        ),
    )
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cm_logistic_regression.png"))
    plt.close()


    print("=== Logistic Regression (with StandardScaler) ===")
    print(classification_report(y_test_enc, y_pred, target_names=class_names))

    acc = accuracy_score(y_test_enc, y_pred)
    f1_macro = f1_score(y_test_enc, y_pred, average="macro")
    f1_weighted = f1_score(y_test_enc, y_pred, average="weighted")
    return acc, f1_macro, f1_weighted


def train_random_forest(X_train, X_test, y_train_enc, y_test_enc, class_names):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train_enc)
    y_pred = rf.predict(X_test)

    cm = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cm_random_forest.png"))
    plt.close()

    plot_feature_importance(
    rf, X_train.columns,"Feature Importance - Random Forest","fi_random_forest.png",)

    
    print("\n=== Random Forest ===")
    print(classification_report(y_test_enc, y_pred, target_names=class_names))

    acc = accuracy_score(y_test_enc, y_pred)
    f1_macro = f1_score(y_test_enc, y_pred, average="macro")
    f1_weighted = f1_score(y_test_enc, y_pred, average="weighted")
    return acc, f1_macro, f1_weighted


def train_xgboost(X_train, X_test, y_train_enc, y_test_enc, class_names, num_classes):
    xgb = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    xgb.fit(X_train, y_train_enc)
    y_pred = xgb.predict(X_test)

    cm = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cm_xgboost.png"))
    plt.close()

    plot_feature_importance(
    xgb,
    X_train.columns,
    "Feature Importance - XGBoost",
    "fi_xgboost.png",
    )

    print("\n=== XGBoost ===")
    print(classification_report(y_test_enc, y_pred, target_names=class_names))

    acc = accuracy_score(y_test_enc, y_pred)
    f1_macro = f1_score(y_test_enc, y_pred, average="macro")
    f1_weighted = f1_score(y_test_enc, y_pred, average="weighted")
    return acc, f1_macro, f1_weighted


def main():
    # 1. Load data
    df = pd.read_csv(r"student-dropout-predictions\data\raw\data.csv", sep=";")
    target_col = "Target"

    # 2. Split features / target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Encode any categorical feature columns (defensive)
    cat_cols = X.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        encoders[col] = enc

    # 4. Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Encode target labels
    y_enc = LabelEncoder()
    y_train_enc = y_enc.fit_transform(y_train)
    y_test_enc = y_enc.transform(y_test)
    class_names = list(y_enc.classes_)
    num_classes = len(class_names)

    # 6. Train three models
    results = {}

    acc, f1_m, f1_w = train_logistic_regression(
        X_train, X_test, y_train_enc, y_test_enc, class_names
    )
    results["Logistic Regression"] = [acc, f1_m, f1_w]

    acc, f1_m, f1_w = train_random_forest(
        X_train, X_test, y_train_enc, y_test_enc, class_names
    )
    results["Random Forest"] = [acc, f1_m, f1_w]

    acc, f1_m, f1_w = train_xgboost(
        X_train, X_test, y_train_enc, y_test_enc, class_names, num_classes
    )
    results["XGBoost"] = [acc, f1_m, f1_w]

    # 7. Comparison table
    results_df = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["Accuracy", "F1_macro", "F1_weighted"],
    )
    print("\n=== Model Comparison (test set) ===")
    print(results_df.round(4))

    # Optionally save to CSV for the paper
    results_df.to_csv("student-dropout-predictions/reports/baseline_results.csv", index=True)

    


if __name__ == "__main__":
    main()
