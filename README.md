# 🧠 IQI-GWO Feature Selection for Image Classification

<p align="center">
  🚀 <b>Optimizing Image Classification using Intelligent Feature Selection</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg">
  <img src="https://img.shields.io/badge/Algorithm-IQI--GWO-green">
  <img src="https://img.shields.io/badge/ML-RandomForest%20%7C%20SVM%20%7C%20MLP-orange">
  <img src="https://img.shields.io/badge/Status-Completed-success">
</p>

---

## 📌 Overview

This project presents an advanced image classification pipeline where **IQI-GWO (Improved Quantum-inspired Grey Wolf Optimizer)** is used for **feature selection**, followed by machine learning models for classification.

The objective is to:

* ✅ Improve classification accuracy
* ✅ Reduce feature dimensionality
* ✅ Remove noisy and irrelevant features

---

## 🧩 Workflow

```text
Image
 ↓
Color Histogram (512 features)
 ↓
IQI-GWO Optimization
 ↓
Optimal Feature Subset
 ↓
Classifier (RF / SVM / MLP)
 ↓
Prediction
```

---

## ⚙️ Core Components

### 📊 Feature Extraction

* RGB Color Histogram (512-dimensional feature vector)

### 🧠 Optimization (IQI-GWO)

* Grey Wolf Optimizer (GWO)
* Quantum-inspired update mechanism
* Sigmoid-based binary conversion
* Mutation for diversity

### 🤖 Classification Models

* Random Forest (Primary)
* Support Vector Machine (Baseline)
* Multi-Layer Perceptron

---

## 📊 Results

| Data Set           | Baseline Accuracy | Accuracy after IQI-GWO |
| ------------------ | ----------------- | ---------------------- |
| LC25000            | 96.3 %            | 99.70 ± 0.09 %         |
| Plant-type-Dataset | 72.45 %           | 97.63 ± 0.02 %         |
| IIITDMJ_Smoke      | 89.23 %           | 96.25 ± 0.04 %         |

---

## 🧠 Fitness Function

$$
Fitness = Accuracy - \lambda \times \frac{Selected\ Features}{Total\ Features}
$$

✔ Maximizes accuracy
✔ Minimizes number of features

---

## 🔬 Key Insights

* 🚀 Significant accuracy improvement after feature selection
* 🧹 Removes redundant and noisy features
* 🎯 Better generalization performance
* ⚡ Reduced computational complexity

---

## ⚠️ Important Notes

* IQI-GWO is used **only for feature selection**, not classification
* Validation set is used during optimization
* Test set is strictly used for **final evaluation only** (avoids data leakage)

---

## 📂 Project Files

* `IQI_GWO.py` → Optimization algorithm
* `Train_Classifier.py` → Model training
* `Evaluate_Model.py` → Performance evaluation
* `Load_DataSet.py` → Dataset loading
* `split_data.py` → Data splitting
* `main.py` → Complete pipeline
* `config_used.json` → Configuration file
* `unzip.py` → Dataset extraction

---

## 📬 Contact

📧 **[ballasandeepsankar@gmail.com](mailto:ballasandeepsankar@gmail.com)**

---

## 🌟 Support

If you found this project useful:

⭐ Star this repository
🍴 Fork it
📢 Share it

---

## 🔮 Future Improvements

* Integrate Deep Learning (CNNs)
* Try hybrid optimization algorithms
* Experiment with advanced feature extraction methods

---

<p align="center">
  💡 <b>"Better features → Better learning → Better results"</b>
</p>
