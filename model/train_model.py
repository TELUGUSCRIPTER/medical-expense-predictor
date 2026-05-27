"""
Train and compare Linear Regression vs Random Forest for insurance prediction.
Saves both models and a comparison report.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'insurance.csv')
MODEL_DIR = os.path.dirname(__file__)


def load_and_preprocess():
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df['region'] = df['region'].map({
        'southwest': 0, 'southeast': 1,
        'northwest': 2, 'northeast': 3,
        'north': 4, 'south': 5,
        'east': 6, 'west': 7, 'central': 8
    })
    return df


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'mae': round(mean_absolute_error(y_test, y_pred), 2),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        'r2': round(r2_score(y_test, y_pred), 4),
        'r2_pct': round(r2_score(y_test, y_pred) * 100, 2),
    }


def train_and_compare(df):
    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_test, y_test)
    lr_coefs = {name: round(coef, 2) for name, coef in zip(X.columns, lr.coef_)}
    lr_coefs['intercept'] = round(lr.intercept_, 2)

    # --- Decision Tree ---
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_metrics = evaluate_model(dt, X_test, y_test)

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)
    rf_importances = {name: round(imp, 4) for name, imp in zip(X.columns, rf.feature_importances_)}

    # --- Logistic Regression ---
    # Create a binary target for Logistic Regression (e.g., above median charges)
    median_charge = y.median()
    y_train_class = (y_train > median_charge).astype(int)
    y_test_class = (y_test > median_charge).astype(int)
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train_class)
    y_pred_class = log_reg.predict(X_test)
    log_reg_accuracy = accuracy_score(y_test_class, y_pred_class)
    log_reg_metrics = {
        'accuracy': round(log_reg_accuracy, 4),
        'accuracy_pct': round(log_reg_accuracy * 100, 2)
    }

    # Print comparison
    print(f"\n{'='*75}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Metric':<20} {'Linear Reg':>14} {'Decision Tree':>15} {'Random Forest':>14}")
    print(f"  {'-'*68}")
    print(f"  {'MAE ($)':<20} {lr_metrics['mae']:>14,.2f} {dt_metrics['mae']:>15,.2f} {rf_metrics['mae']:>14,.2f}")
    print(f"  {'RMSE ($)':<20} {lr_metrics['rmse']:>14,.2f} {dt_metrics['rmse']:>15,.2f} {rf_metrics['rmse']:>14,.2f}")
    print(f"  {'R² Score':<20} {lr_metrics['r2']:>14.4f} {dt_metrics['r2']:>15.4f} {rf_metrics['r2']:>14.4f}")
    print(f"  {'Accuracy %':<20} {lr_metrics['r2_pct']:>13.2f}% {dt_metrics['r2_pct']:>14.2f}% {rf_metrics['r2_pct']:>13.2f}%")
    print(f"{'='*75}")
    print(f"\n  Logistic Regression (Classification > Median): Accuracy {log_reg_metrics['accuracy_pct']}%")
    print(f"{'='*75}")

    winner = 'Random Forest' if rf_metrics['r2'] > lr_metrics['r2'] and rf_metrics['r2'] > dt_metrics['r2'] else ('Decision Tree' if dt_metrics['r2'] > lr_metrics['r2'] else 'Linear Regression')
    
    # Save comparison report
    comparison = {
        'linear_regression': {**lr_metrics, 'coefficients': lr_coefs},
        'decision_tree': dt_metrics,
        'random_forest': {**rf_metrics, 'feature_importances': rf_importances},
        'logistic_regression': log_reg_metrics,
        'winner': winner,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
    with open(os.path.join(MODEL_DIR, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    return lr, dt, rf, log_reg, comparison


def save_models(lr, dt, rf, log_reg):
    with open(os.path.join(MODEL_DIR, 'linear_model.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(MODEL_DIR, 'dt_model.pkl'), 'wb') as f:
        pickle.dump(dt, f)
    with open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(MODEL_DIR, 'log_model.pkl'), 'wb') as f:
        pickle.dump(log_reg, f)
    # Save RF as default (better accuracy)
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    print(f"\nModels saved to: {MODEL_DIR}/")


if __name__ == '__main__':
    print("=" * 55)
    print("  Medical Insurance Prediction - Model Training")
    print("=" * 55)
    df = load_and_preprocess()
    lr, dt, rf, log_reg, comparison = train_and_compare(df)
    save_models(lr, dt, rf, log_reg)
    print("Training complete!\n")
