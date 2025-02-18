# 🚗 Lane-Changing Decision Intention Prediction for Intelligent Driving

## 📌 Project Overview
This project predicts **lane-changing intentions** of surrounding vehicles using **machine learning** and **driver behavior analysis**. By analyzing vehicle trajectory data, the model determines whether a driver intends to **change lanes (1)** or **stay in the current lane (0)**. The system integrates **behavioral clustering, ensemble modeling, and misclassification filtering** to enhance accuracy.

Understanding lane-change behavior improves **autonomous vehicle navigation**, **reduces collision risks**, and **optimizes traffic flow**. The model processes real-time driving data and adapts to different driving styles, supporting advanced driver-assistance systems (ADAS).

---

## 🚀 Key Features
✅ **Driver Behavior Analysis** using **Gaussian Mixture Model (GMM)**  
✅ **Machine Learning-Based Lane Change Prediction** using **LightGBM**  
✅ **Class Imbalance Handling** with **SMOTE & Undersampling**  
✅ **Stacked Model (LightGBM + XGBoost)** for improved accuracy  
✅ **Misclassification Filtering** to remove incorrect predictions  
✅ **Hyperparameter Optimization** using **RandomizedSearchCV**  
✅ **Real-Time Lane-Change Prediction Capability**  

---

## 📊 Dataset & Feature Engineering
- **Features Used:**
  - **Velocity (`v_Vel`)**
  - **Acceleration (`v_Acc`)**
  - **Time Headway (`THW`)**
  - **Modified Time-to-Collision (`MTTC`)**
  - **Jerk (Rate of Acceleration Change)**
- **Driver Behavior Classification:**
  - **Cautious** (smooth & defensive)
  - **Neutral** (moderate risk-taking)
  - **Aggressive** (frequent lane changes, high acceleration)
  
---

## 🏗 Model Training Pipeline
1. **Data Preprocessing & Feature Engineering**
2. **Driver Behavior Clustering (GMM)**
3. **Handle Class Imbalance (SMOTE & Undersampling)**
4. **Train LightGBM & Optimize Hyperparameters**
5. **Stack LightGBM + XGBoost for Better Generalization**
6. **Filter Out Misclassified Samples & Retrain**
7. **Evaluate Model Performance**

---

## 📈 Performance Metrics
- **Accuracy Score** → Measures overall correctness
- **ROC AUC Score** → Evaluates lane-change classification quality
- **Confusion Matrix** → Highlights misclassifications
- **Precision-Recall** → Ensures balance between false positives & negatives

---
