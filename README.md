# Autonomous-Lane-changing-vechicle-prediction-model
Lane-Changing Decision Intention Prediction for Intelligent Driving
📌 Project Overview
This project aims to predict the lane-changing intentions of surrounding vehicles, a crucial component of intelligent driving systems and autonomous vehicles. By analyzing vehicle trajectory data and driver behavior, the model determines whether a driver intends to change lanes (1) or stay in the current lane (0). The system integrates machine learning, behavioral clustering, and ensemble modeling to improve prediction accuracy and robustness.

Understanding lane-change behavior enhances autonomous vehicle navigation, reduces collision risks, and improves traffic flow efficiency. The model is designed to process real-time driving data, adapt to different driving styles, and make reliable predictions to support advanced driver-assistance systems (ADAS).

🚗 Key Components of the Project
1️⃣ Data Preprocessing & Feature Engineering
The dataset contains vehicle trajectory data, including features like:
Velocity (v_Vel)
Acceleration (v_Acc)
Time Headway (THW)
Modified Time-to-Collision (MTTC)
Jerk (Rate of acceleration change)
Driver Behavior Analysis:
Uses Gaussian Mixture Model (GMM) to classify drivers into:
Cautious (smooth and defensive driving)
Neutral (moderate risk-taking)
Aggressive (high acceleration and frequent lane changes)
Behavioral clustering adds contextual information to the lane-change prediction model.
2️⃣ Handling Class Imbalance
Since lane-changing events occur less frequently than lane-keeping, class imbalance is addressed using:
✅ SMOTE (Synthetic Minority Over-sampling Technique) – Generates synthetic lane-change samples to balance the dataset.
✅ Random Undersampling – Reduces excessive lane-keeping samples to prevent bias.

3️⃣ Model Training & Hyperparameter Optimization
Trains LightGBM, a gradient boosting decision tree algorithm, to classify lane-change intentions.
Performs hyperparameter tuning using RandomizedSearchCV to optimize:
Learning rates
Tree depths
Feature selection
Stacks LightGBM and XGBoost models for improved accuracy and generalization.
4️⃣ Misclassification Filtering & Model Retraining
Identifies test cases where the model predicts incorrectly.
Removes these misclassified samples to clean the dataset.
Retrains the model using only correctly classified samples, improving model robustness.
📊 Model Evaluation & Performance Metrics
Assesses model effectiveness using:
✅ Accuracy Score – Measures overall correctness.
✅ ROC AUC Score – Evaluates true positive vs. false positive rates.
✅ Confusion Matrix – Provides insights into classification errors.
✅ Precision-Recall Metrics – Determines how well the model distinguishes lane changes.
🎯 Benefits & Real-World Applications
🚀 Autonomous Driving Enhancement → Enables self-driving cars to anticipate lane-change maneuvers.
🚀 Collision Avoidance → Helps autonomous systems react to sudden lane changes.
🚀 Traffic Flow Optimization → Supports intelligent transportation planning and highway safety.
🚀 Adaptive Driver Assistance → Integrates into ADAS for lane-change warnings and automated maneuvering.

🛠 Future Improvements & Extensions
🔹 Incorporate Real-Time Sensor Data → Combine camera, LiDAR, and radar for richer feature sets.
🔹 Deep Learning Integration → Use LSTMs or Transformers for time-series lane-change prediction.
🔹 Expand to Multi-Lane Scenarios → Improve predictions in congested, multi-lane highway environments.
