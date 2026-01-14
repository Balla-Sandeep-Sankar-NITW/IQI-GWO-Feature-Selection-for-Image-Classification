
# CLASSIFIER 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def train_classifier(X, y, method="RandomForest"):
    print(f"🎓 Training {method} classifier...")
    if method == "RandomForest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    elif method == "SVC":
        clf = SVC(kernel="rbf", probability=True, random_state=42)
    elif method == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {method}")
    clf.fit(X, y)
    return clf