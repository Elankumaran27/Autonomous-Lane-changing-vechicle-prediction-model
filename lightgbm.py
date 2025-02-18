# Import libraries
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb

# Step 1: Load dataset
file_path = '/kaggle/input/feature/feature_engineered_data (1).csv'
data = pd.read_csv(file_path)

# Step 2: Driver Behavior Analysis (GMM Clustering)
behavior_features = ['v_Vel', 'v_Acc', 'THW', 'MTTC', 'Jerk']
gmm = GaussianMixture(n_components=3, random_state=42)
data['Driving_Style'] = gmm.fit_predict(data[behavior_features])

# Step 3: Features & Target
X = data.drop(columns=['Lane_Change'])
y = data['Lane_Change']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Hyperparameter Grid (Optimized for Speed)
param_grid = {
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
    'max_depth': [5, 10],
    'n_estimators': [100, 200],
    'min_child_samples': [20, 50],
    'feature_fraction': [0.9, 1.0]
}

# Step 7: LightGBM Classifier
lgb_model = lgb.LGBMClassifier(objective='binary', boosting_type='gbdt', random_state=42, n_jobs=-1)

# Step 8: Randomized Search (Faster Tuning)
grid_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    scoring='roc_auc',
    n_iter=10,  # Reduced iterations
    cv=2,       # Reduced cross-validation folds
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Starting hyperparameter tuning...")
grid_search.fit(X_train_scaled, y_train)
print("Hyperparameter tuning complete.")

# Step 9: Best Parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Step 10: Final Model Training
best_model = lgb.LGBMClassifier(**best_params, objective='binary', boosting_type='gbdt', random_state=42)
best_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(10)]
)

# Step 11: Evaluation
y_pred = best_model.predict(X_test_scaled)
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

# Step 12: Save the Model
best_model.booster_.save_model('/kaggle/working/optimized_lane_change_model.txt')
