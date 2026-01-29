# 🩺 Breast Cancer Prediction using Machine Learning

## 📌 Project Overview
Breast cancer is one of the leading causes of mortality among women worldwide, and early detection plays a crucial role in improving survival rates.  
This project presents an end-to-end **machine learning–based diagnostic system** to classify breast tumours as **benign** or **malignant** using clinical features from the **Wisconsin Breast Cancer Diagnostic (WBCD) dataset**.

The project was developed as part of a **Python and Machine Learning Internship** and focuses on building an accurate, interpretable, and reliable prediction pipeline suitable for healthcare decision-support systems.

---

## 👩‍💻 Team Members
- **Pratigya Sachdeva** (15501012024)  
- **Neha Binu** (13201012024)

---

## 🎯 Objectives
- Understand and analyse a real-world medical dataset  
- Apply complete data preprocessing and exploratory data analysis  
- Train and evaluate multiple machine learning classification models  
- Perform feature selection and dimensionality reduction  
- Improve model performance through optimization techniques  
- Ensure transparency and interpretability using Explainable AI (XAI)

---

## 📊 Dataset
- **Dataset Name:** Wisconsin Breast Cancer Diagnostic (WBCD) Dataset  
- **Source:** UCI Machine Learning Repository  
- **Features:** 30 numerical diagnostic features  
- **Target Classes:**
  - `M` → Malignant  
  - `B` → Benign  

The dataset contains measurements computed from digitized images of breast mass cell nuclei.

---

## ⚙️ Tools & Technologies Used

### Programming & Environment
- Python 3.x  
- Google Colab  

### Libraries
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- SHAP (Explainable AI)

---

## 🧠 Methodology

### 1. Data Preprocessing
- Removal of non-informative columns  
- Handling missing values and duplicates  
- Outlier detection using IQR  
- Skewness correction using log transformations  
- Feature scaling using `StandardScaler`  
- Multicollinearity reduction using correlation analysis and VIF  

---

### 2. Exploratory Data Analysis (EDA)
- Feature distribution analysis (histograms & KDE plots)  
- Pairwise feature relationships  
- Correlation heatmaps to identify highly correlated attributes  

---

### 3. Model Development
The following supervised learning models were implemented and evaluated:
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Support Vector Machine (SVM – RBF kernel)

Each model was trained using an 80/20 stratified train–test split and evaluated using standard performance metrics.

---

### 4. Feature Selection & Dimensionality Reduction
- **Principal Component Analysis (PCA)**  
- Recursive Feature Elimination (RFE)  
- Chi-Square Test  

PCA reduced the feature space from 30 to 9 components while preserving ~95% variance.

---

### 5. Ensemble & Optimization Techniques
- Voting Classifier  
- Stacking Classifier  
- Genetic Algorithm (GA)  
- Particle Swarm Optimization (PSO)  

These techniques were explored to improve model generalization and robustness.

---

### 6. Explainable AI (XAI)
- SHAP (SHapley Additive Explanations) was used to interpret model predictions  
- Key influential features included:
  - `concavity_worst`
  - `concave_points_mean`
  - `radius_mean`
  - `texture_mean`

This enhanced transparency and trust in the predictive system.

---

## 📈 Results & Performance

| Model | Accuracy |
|------|----------|
| Logistic Regression | ~95.6% |
| Random Forest | ~94.7% |
| Gradient Boosting | ~96.5% |
| SVM (RBF Kernel) | ~97.4% |
| **PCA + SVM (Best Model)** | **≈ 99.12%** |

✅ **Best Performing Model:** PCA + SVM  
✅ **Final Accuracy:** ≈ **99.12%**


▶️ How to Run the Project
	1.	Clone the repository:
       git clone https://github.com/your-username/Breast-Cancer-Prediction-ML
  
  2.	Open the notebook:
       Breast_Cancer_Prediction.ipynb

📌 Conclusion
This project demonstrates how structured data preprocessing, robust machine learning models, and explainable AI techniques can be combined to build a highly accurate and interpretable breast cancer prediction system.
The PCA-enhanced SVM model achieved superior performance and highlights the importance of dimensionality reduction in medical datasets.

The project contributes to the broader goal of applying machine learning in healthcare for early diagnosis and clinical decision support.

🔗 References
	•	UCI Machine Learning Repository – Breast Cancer Dataset
	•	Lundberg & Lee (2017) – SHAP Explainability
	•	World Health Organization – Breast Cancer Factsheet
