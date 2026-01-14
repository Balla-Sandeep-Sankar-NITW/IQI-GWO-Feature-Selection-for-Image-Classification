import os
import json
import argparse
import warnings
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from Load_DataSet import load_dataset_from_path
from IQI_GWO import IQI_GWO
from Train_Classifier import train_classifier
from Evaluate_Model import evaluate_model

warnings.filterwarnings("ignore")
np.random.seed(42)


CONFIG = {
    "image_size": (128, 128),
    "population_size": 10,
    "max_iters": 50,
    "quantum_rate": 0.25,
    "mutation_rate": 0.05,
    "penalty_coef": 0.001,
    "classifier_method": "RandomForest",
    "use_scaling": True
}


def main():
    parser = argparse.ArgumentParser(description="IQI-GWO Image Classification Pipeline")
    parser.add_argument("--train", required=True, help="Path to training dataset")
    parser.add_argument("--val", required=True, help="Path to validation dataset")
    parser.add_argument("--test", required=True, help="Path to test dataset")

    args = parser.parse_args()

    dataset_train_path = os.path.abspath(args.train)
    dataset_val_path = os.path.abspath(args.val)
    dataset_test_path = os.path.abspath(args.test)

    print("\n🚀 IQI-GWO Lightweight Image Classification (Train/Val/Test)")
    print("===========================================================")
    print(f"Train Path: {dataset_train_path}")
    print(f"Val Path:   {dataset_val_path}")
    print(f"Test Path:  {dataset_test_path}")


    # ✅ Load datasets
    X_train, y_train, classes = load_dataset_from_path(dataset_train_path)
    X_val, y_val, _ = load_dataset_from_path(dataset_val_path)
    X_test, y_test, _ = load_dataset_from_path(dataset_test_path)

    # ⚖️ Baseline (SVM on all features)
    print("\n⚖️ Running baseline SVM...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    base = SVC(kernel="rbf", probability=True, random_state=42)
    base.fit(X_train_s, y_train)
    base_metrics = evaluate_model(base, X_test_s, y_test, classes)

    # ✨ Feature Selection (IQI-GWO)
    print("\n✨ Feature Selection with IQI-GWO using Validation Set...")
    gwo = IQI_GWO(
        X_train.shape[1],
        CONFIG["population_size"],
        CONFIG["max_iters"],
        CONFIG["quantum_rate"],
        CONFIG["mutation_rate"],
        CONFIG["penalty_coef"]
    )

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    mask, hist = gwo.optimize(X_train, y_train, X_val, y_val, rf)

    np.save("selected_features.npy", np.where(mask == 1)[0])
    print(f"Selected {mask.sum()}/{len(mask)} features")

    X_train_sel = X_train[:, mask == 1]
    X_test_sel = X_test[:, mask == 1]

    X_train_sel = scaler.fit_transform(X_train_sel)
    X_test_sel = scaler.transform(X_test_sel)

    clf = train_classifier(X_train_sel, y_train, CONFIG["classifier_method"])
    joblib.dump(clf, "trained_model.joblib")

    res = evaluate_model(clf, X_test_sel, y_test, classes)

    print(f"\n✅ Done! Baseline Acc: {(base_metrics['accuracy']*100):.4f} %, After IQI-GWO: {(res['accuracy']*100):.4f} %")


if __name__ == "__main__":
    main()
