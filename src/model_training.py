"""
Model training module for German Credit Dataset.
Implements multiple ML algorithms with MLflow tracking and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from data_preprocessing import GermanCreditPreprocessor


class CreditRiskModelTrainer:
    """
    Model trainer for credit risk prediction.
    Supports multiple algorithms with MLflow experiment tracking.
    """

    def __init__(self, experiment_name="german_credit_risk"):
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.cv_folds = 5

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

    def prepare_data(self):
        """Load and preprocess data."""
        preprocessor = GermanCreditPreprocessor()
        df = preprocessor.load_data()
        X, y, feature_names = preprocessor.fit_transform(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        self.feature_names = feature_names
        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression with hyperparameter tuning."""
        print("Training Logistic Regression...")

        with mlflow.start_run(run_name="logistic_regression"):
            # Hyperparameter grid
            param_grid = {
                "C": [0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "max_iter": [1000],
            }

            # Grid search with cross-validation
            lr = LogisticRegression(random_state=42)
            grid_search = GridSearchCV(
                lr,
                param_grid,
                cv=self.cv_folds,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            # Best model
            best_lr = grid_search.best_estimator_

            # Predictions
            y_pred = best_lr.predict(X_test)
            y_pred_proba = best_lr.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_lr, "model")

            # Cross-validation score
            cv_scores = cross_val_score(
                best_lr, X_train, y_train, cv=self.cv_folds, scoring="roc_auc"
            )
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())

            self.models["logistic_regression"] = {
                "model": best_lr,
                "metrics": metrics,
                "cv_scores": cv_scores,
            }

            print(f"Logistic Regression - Test AUC: {metrics['roc_auc']:.4f}")
            return best_lr, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest with hyperparameter tuning."""
        print("Training Random Forest...")

        with mlflow.start_run(run_name="random_forest"):
            # Hyperparameter grid
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            }

            # Grid search with cross-validation
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf,
                param_grid,
                cv=self.cv_folds,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            # Best model
            best_rf = grid_search.best_estimator_

            # Predictions
            y_pred = best_rf.predict(X_test)
            y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_rf, "model")

            # Feature importance
            feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": best_rf.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            # Log top 10 feature importances
            for i, row in feature_importance.head(10).iterrows():
                mlflow.log_metric(
                    f"feature_importance_{row['feature']}", row["importance"]
                )

            # Cross-validation score
            cv_scores = cross_val_score(
                best_rf, X_train, y_train, cv=self.cv_folds, scoring="roc_auc"
            )
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())

            self.models["random_forest"] = {
                "model": best_rf,
                "metrics": metrics,
                "cv_scores": cv_scores,
                "feature_importance": feature_importance,
            }

            print(f"Random Forest - Test AUC: {metrics['roc_auc']:.4f}")
            return best_rf, metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost with hyperparameter tuning."""
        print("Training XGBoost...")

        with mlflow.start_run(run_name="xgboost"):
            # Hyperparameter grid
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }

            # Grid search with cross-validation
            xgb_model = xgb.XGBClassifier(
                random_state=42, eval_metric="logloss", use_label_encoder=False
            )
            grid_search = GridSearchCV(
                xgb_model,
                param_grid,
                cv=self.cv_folds,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            # Best model
            best_xgb = grid_search.best_estimator_

            # Predictions
            y_pred = best_xgb.predict(X_test)
            y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(best_xgb, "model")

            # Feature importance
            feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": best_xgb.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            # Log top 10 feature importances
            for i, row in feature_importance.head(10).iterrows():
                mlflow.log_metric(
                    f"feature_importance_{row['feature']}", row["importance"]
                )

            # Cross-validation score
            cv_scores = cross_val_score(
                best_xgb, X_train, y_train, cv=self.cv_folds, scoring="roc_auc"
            )
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())

            self.models["xgboost"] = {
                "model": best_xgb,
                "metrics": metrics,
                "cv_scores": cv_scores,
                "feature_importance": feature_importance,
            }

            print(f"XGBoost - Test AUC: {metrics['roc_auc']:.4f}")
            return best_xgb, metrics

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }

    def train_all_models(self):
        """Train all models and compare performance."""
        print("Starting model training pipeline...")

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Train models
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)

        # Find best model
        self._select_best_model()

        # Save best model
        self._save_best_model()

        return self.models

    def _select_best_model(self):
        """Select the best model based on ROC AUC score."""
        best_auc = 0
        best_model_name = None

        print("\n=== MODEL COMPARISON ===")
        for model_name, model_info in self.models.items():
            auc = model_info["metrics"]["roc_auc"]
            cv_auc = model_info["cv_scores"].mean()
            print(f"{model_name.upper()}:")
            print(f"  Test AUC: {auc:.4f}")
            print(f"  CV AUC: {cv_auc:.4f} ± {model_info['cv_scores'].std():.4f}")
            print(f"  Precision: {model_info['metrics']['precision']:.4f}")
            print(f"  Recall: {model_info['metrics']['recall']:.4f}")
            print(f"  F1-Score: {model_info['metrics']['f1_score']:.4f}")
            print()

            if auc > best_auc:
                best_auc = auc
                best_model_name = model_name

        self.best_model = self.models[best_model_name]["model"]
        self.best_score = best_auc

        print(f"BEST MODEL: {best_model_name.upper()} (AUC: {best_auc:.4f})")

    def _save_best_model(self):
        """Save the best model to disk."""
        if self.best_model is not None:
            os.makedirs("../models", exist_ok=True)
            model_path = f"../models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(self.best_model, model_path)
            print(f"Best model saved to: {model_path}")

    def generate_model_report(self):
        """Generate a comprehensive model performance report."""
        report = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "models_trained": len(self.models),
            "best_model_auc": self.best_score,
            "model_comparison": {},
        }

        for model_name, model_info in self.models.items():
            report["model_comparison"][model_name] = {
                "test_metrics": model_info["metrics"],
                "cv_auc_mean": model_info["cv_scores"].mean(),
                "cv_auc_std": model_info["cv_scores"].std(),
            }

        return report


def main():
    """Main function to run the complete model training pipeline."""
    print("=== GERMAN CREDIT RISK MODEL TRAINING ===")

    # Initialize trainer
    trainer = CreditRiskModelTrainer()

    # Train all models
    models = trainer.train_all_models()

    # Generate report
    report = trainer.generate_model_report()

    print("\n=== TRAINING COMPLETED ===")
    print(f"Best model AUC: {report['best_model_auc']:.4f}")
    print("Check MLflow UI for detailed experiment tracking")

    return trainer, models, report


if __name__ == "__main__":
    trainer, models, report = main()
