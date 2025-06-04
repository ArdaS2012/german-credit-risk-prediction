"""
Model training module for German Credit Dataset.

This module implements a comprehensive machine learning pipeline for training and 
evaluating credit risk prediction models. It supports multiple algorithms with 
automated hyperparameter tuning, cross-validation, and MLflow experiment tracking.

Key Features:
- Multiple ML algorithms: Logistic Regression, Random Forest, XGBoost
- Automated hyperparameter optimization using GridSearchCV
- Comprehensive model evaluation with multiple metrics
- MLflow integration for experiment tracking and reproducibility
- Cross-validation for robust performance estimation
- Automatic model selection based on ROC-AUC scores
- Model persistence for production deployment

Supported Algorithms:
- Logistic Regression: Linear probabilistic classifier with regularization
- Random Forest: Ensemble method with bootstrap aggregating
- XGBoost: Gradient boosting with advanced optimization techniques

Author: German Credit Risk Prediction System
Version: 1.0.0
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

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore")

# Import our custom preprocessing module
from data_preprocessing import GermanCreditPreprocessor


class CreditRiskModelTrainer:
    """
    Comprehensive model trainer for credit risk prediction with MLflow tracking.
    
    This class implements a complete machine learning pipeline that trains multiple
    algorithms, performs hyperparameter optimization, evaluates model performance,
    and tracks experiments using MLflow. It automatically selects the best performing
    model based on ROC-AUC score and saves it for production use.
    
    The trainer supports three main algorithms:
    1. Logistic Regression - Fast, interpretable linear model
    2. Random Forest - Robust ensemble method with feature importance
    3. XGBoost - State-of-the-art gradient boosting algorithm
    
    Attributes:
        experiment_name (str): MLflow experiment name for tracking
        models (dict): Dictionary storing trained models and their results
        best_model (object): Best performing model object
        best_score (float): ROC-AUC score of the best model
        cv_folds (int): Number of cross-validation folds
        feature_names (list): Names of input features
    
    Example:
        >>> trainer = CreditRiskModelTrainer("credit_risk_experiment")
        >>> trainer.train_all_models()
        >>> print(f"Best model: {trainer.best_model}")
        >>> print(f"Best score: {trainer.best_score:.4f}")
    """

    def __init__(self, experiment_name="german_credit_risk"):
        """
        Initialize the Credit Risk Model Trainer.
        
        Sets up MLflow experiment tracking, initializes storage for models and results,
        and configures cross-validation parameters.
        
        Args:
            experiment_name (str): Name for MLflow experiment tracking.
                                 Defaults to "german_credit_risk"
        
        Side Effects:
            - Creates or sets MLflow experiment
            - Initializes empty model storage and tracking variables
        """
        # MLflow experiment configuration
        self.experiment_name = experiment_name
        
        # Model storage and tracking
        self.models = {}           # Stores trained models and their metadata
        self.best_model = None     # Reference to best performing model
        self.best_score = 0        # ROC-AUC score of best model
        self.cv_folds = 5          # Number of cross-validation folds
        self.feature_names = None  # Will be set during data preparation

        # Set up MLflow experiment tracking
        # This allows us to track parameters, metrics, and artifacts
        mlflow.set_experiment(experiment_name)
        print(f"🔬 MLflow experiment initialized: {experiment_name}")

    def prepare_data(self):
        """
        Load and preprocess the German Credit Dataset for model training.
        
        This method handles the complete data preparation workflow:
        1. Loads raw data using GermanCreditPreprocessor
        2. Applies preprocessing transformations (encoding, scaling)
        3. Splits data into training and testing sets
        4. Stores feature names for later use
        
        Returns:
            tuple: A 4-element tuple containing:
                - X_train (np.ndarray): Training feature matrix
                - X_test (np.ndarray): Testing feature matrix
                - y_train (np.ndarray): Training target values
                - y_test (np.ndarray): Testing target values
        
        Side Effects:
            - Sets self.feature_names for model interpretability
        
        Example:
            >>> trainer = CreditRiskModelTrainer()
            >>> X_train, X_test, y_train, y_test = trainer.prepare_data()
            >>> print(f"Training samples: {len(X_train)}")
        """
        print("📊 Preparing data for model training...")
        
        # Initialize and use our custom preprocessor
        preprocessor = GermanCreditPreprocessor()
        
        # Load the raw dataset
        print("   Loading German Credit Dataset...")
        df = preprocessor.load_data()
        
        # Apply preprocessing transformations
        print("   Applying preprocessing pipeline...")
        X, y, feature_names = preprocessor.fit_transform(df)
        
        # Store feature names for model interpretation and logging
        self.feature_names = feature_names
        
        # Split data into training and testing sets
        print("   Splitting data for training and evaluation...")
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        print(f"✅ Data preparation completed!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Features: {len(feature_names)}")
        
        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train Logistic Regression model with comprehensive hyperparameter tuning.
        
        Logistic Regression is a linear model that uses the logistic function to
        model the probability of binary outcomes. It's fast, interpretable, and
        works well as a baseline model for credit risk prediction.
        
        This method performs:
        1. Hyperparameter grid search with cross-validation
        2. Model fitting on best parameters
        3. Performance evaluation on test set
        4. MLflow logging of parameters, metrics, and model
        5. Cross-validation score calculation
        
        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training target values
            X_test (np.ndarray): Testing feature matrix
            y_test (np.ndarray): Testing target values
        
        Returns:
            tuple: (best_model, metrics_dict) containing the trained model and evaluation metrics
        
        Hyperparameters Tuned:
            - C: Regularization strength (0.1, 1.0, 10.0, 100.0)
            - penalty: Regularization type ('l1', 'l2')
            - solver: Optimization algorithm ('liblinear', 'saga')
            - max_iter: Maximum iterations (1000)
        """
        print("🔮 Training Logistic Regression with hyperparameter tuning...")

        # Start MLflow run for this specific model
        with mlflow.start_run(run_name="logistic_regression"):
            
            # Define hyperparameter grid for comprehensive search
            # C: Inverse of regularization strength (smaller = more regularization)
            # penalty: Type of regularization (L1 for feature selection, L2 for coefficient shrinkage)
            # solver: Algorithm for optimization (liblinear good for small datasets)
            param_grid = {
                "C": [0.1, 1.0, 10.0, 100.0],      # Regularization strength
                "penalty": ["l1", "l2"],            # Regularization type
                "solver": ["liblinear", "saga"],    # Optimization algorithm
                "max_iter": [1000],                 # Convergence iterations
            }

            # Initialize base model with fixed random state for reproducibility
            lr = LogisticRegression(random_state=42)
            
            # Perform grid search with cross-validation
            # StratifiedKFold preserves class distribution in each fold
            # ROC-AUC scoring is ideal for binary classification with class imbalance
            print("   Performing grid search with cross-validation...")
            grid_search = GridSearchCV(
                lr,
                param_grid,
                cv=self.cv_folds,         # 5-fold cross-validation
                scoring="roc_auc",        # Optimize for ROC-AUC
                n_jobs=-1,                # Use all available cores
                verbose=1,                # Show progress
            )
            
            # Fit grid search to find best hyperparameters
            grid_search.fit(X_train, y_train)

            # Extract best model with optimal hyperparameters
            best_lr = grid_search.best_estimator_
            print(f"   Best parameters: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")

            # Make predictions on test set for evaluation
            y_pred = best_lr.predict(X_test)
            y_pred_proba = best_lr.predict_proba(X_test)[:, 1]  # Probability of positive class

            # Calculate comprehensive evaluation metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Log hyperparameters and metrics to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Log the trained model for later retrieval
            mlflow.sklearn.log_model(best_lr, "model")

            # Perform additional cross-validation for robust performance estimation
            print("   Performing cross-validation for robust evaluation...")
            cv_scores = cross_val_score(
                best_lr, X_train, y_train, 
                cv=self.cv_folds, 
                scoring="roc_auc"
            )
            
            # Log cross-validation statistics
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())

            # Store model results for comparison
            self.models["logistic_regression"] = {
                "model": best_lr,
                "metrics": metrics,
                "cv_scores": cv_scores,
                "best_params": grid_search.best_params_
            }

            print(f"✅ Logistic Regression training completed!")
            print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return best_lr, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest model with extensive hyperparameter optimization.
        
        Random Forest is an ensemble method that builds multiple decision trees
        and combines their predictions through voting. It's robust to overfitting,
        handles feature interactions well, and provides feature importance scores.
        
        This method performs:
        1. Comprehensive hyperparameter grid search
        2. Model training with best parameters
        3. Feature importance analysis and logging
        4. Performance evaluation with multiple metrics
        5. MLflow experiment tracking
        
        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training target values
            X_test (np.ndarray): Testing feature matrix
            y_test (np.ndarray): Testing target values
        
        Returns:
            tuple: (best_model, metrics_dict) containing the trained model and evaluation metrics
        
        Hyperparameters Tuned:
            - n_estimators: Number of trees (100, 200, 300)
            - max_depth: Maximum tree depth (10, 20, None)
            - min_samples_split: Minimum samples to split a node (2, 5, 10)
            - min_samples_leaf: Minimum samples in leaf nodes (1, 2, 4)
            - max_features: Features considered at each split ('sqrt', 'log2')
        """
        print("🌲 Training Random Forest with hyperparameter tuning...")

        # Start MLflow run for Random Forest
        with mlflow.start_run(run_name="random_forest"):
            
            # Define comprehensive hyperparameter grid
            # n_estimators: More trees generally improve performance but increase computation
            # max_depth: Controls tree complexity (None allows unlimited depth)
            # min_samples_split/leaf: Control overfitting by requiring minimum samples
            # max_features: Controls randomness and reduces correlation between trees
            param_grid = {
                "n_estimators": [100, 200, 300],        # Number of trees in forest
                "max_depth": [10, 20, None],            # Maximum depth of trees
                "min_samples_split": [2, 5, 10],        # Min samples to split internal node
                "min_samples_leaf": [1, 2, 4],          # Min samples in leaf node
                "max_features": ["sqrt", "log2"],       # Features to consider at each split
            }

            # Initialize Random Forest with fixed random state for reproducibility
            rf = RandomForestClassifier(
                random_state=42, 
                n_jobs=-1  # Use all cores for parallel tree building
            )
            
            # Perform grid search with cross-validation
            print("   Performing comprehensive grid search...")
            grid_search = GridSearchCV(
                rf,
                param_grid,
                cv=self.cv_folds,         # Stratified cross-validation
                scoring="roc_auc",        # Optimize for ROC-AUC
                n_jobs=-1,                # Parallel processing
                verbose=1,                # Progress monitoring
            )
            
            # Fit grid search to find optimal hyperparameters
            grid_search.fit(X_train, y_train)

            # Extract best model
            best_rf = grid_search.best_estimator_
            print(f"   Best parameters: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")

            # Generate predictions for evaluation
            y_pred = best_rf.predict(X_test)
            y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

            # Calculate evaluation metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Log parameters and metrics to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_rf, "model")

            # Analyze and log feature importance
            # Random Forest provides natural feature importance through impurity reduction
            print("   Analyzing feature importance...")
            feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": best_rf.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            # Log top 10 most important features to MLflow
            for i, row in feature_importance.head(10).iterrows():
                mlflow.log_metric(
                    f"feature_importance_{row['feature']}", row["importance"]
                )
            
            print("   Top 5 most important features:")
            for i, row in feature_importance.head(5).iterrows():
                print(f"     {row['feature']}: {row['importance']:.4f}")

            # Perform cross-validation for robust evaluation
            print("   Performing cross-validation...")
            cv_scores = cross_val_score(
                best_rf, X_train, y_train, 
                cv=self.cv_folds, 
                scoring="roc_auc"
            )
            
            # Log cross-validation results
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())

            # Store complete model results
            self.models["random_forest"] = {
                "model": best_rf,
                "metrics": metrics,
                "cv_scores": cv_scores,
                "feature_importance": feature_importance,
                "best_params": grid_search.best_params_
            }

            print(f"✅ Random Forest training completed!")
            print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return best_rf, metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost model with advanced hyperparameter optimization.
        
        XGBoost (eXtreme Gradient Boosting) is a highly optimized gradient boosting
        algorithm that often achieves state-of-the-art performance in machine learning
        competitions. It includes advanced features like regularization, feature
        importance, and efficient handling of missing values.
        
        This method performs:
        1. Extensive hyperparameter grid search
        2. Model training with optimal parameters
        3. Feature importance analysis
        4. Comprehensive performance evaluation
        5. MLflow tracking with XGBoost-specific logging
        
        Args:
            X_train (np.ndarray): Training feature matrix
            y_train (np.ndarray): Training target values
            X_test (np.ndarray): Testing feature matrix
            y_test (np.ndarray): Testing target values
        
        Returns:
            tuple: (best_model, metrics_dict) containing the trained model and evaluation metrics
        
        Hyperparameters Tuned:
            - n_estimators: Number of boosting rounds (100, 200, 300)
            - max_depth: Maximum tree depth (3, 6, 9)
            - learning_rate: Step size shrinkage (0.01, 0.1, 0.2)
            - subsample: Fraction of samples used per tree (0.8, 0.9, 1.0)
            - colsample_bytree: Fraction of features used per tree (0.8, 0.9, 1.0)
        """
        print("🚀 Training XGBoost with advanced hyperparameter tuning...")

        # Start MLflow run with XGBoost-specific tracking
        with mlflow.start_run(run_name="xgboost"):
            
            # Define XGBoost hyperparameter grid
            # n_estimators: Number of gradient boosting rounds
            # max_depth: Controls tree complexity and overfitting
            # learning_rate: Controls contribution of each tree (lower = more conservative)
            # subsample: Random sampling of training instances (prevents overfitting)
            # colsample_bytree: Random sampling of features (adds randomness)
            param_grid = {
                "n_estimators": [100, 200, 300],        # Number of boosting rounds
                "max_depth": [3, 6, 9],                 # Maximum tree depth
                "learning_rate": [0.01, 0.1, 0.2],     # Step size shrinkage
                "subsample": [0.8, 0.9, 1.0],          # Sample fraction per tree
                "colsample_bytree": [0.8, 0.9, 1.0],   # Feature fraction per tree
            }

            # Initialize XGBoost classifier with optimized settings
            xgb_model = xgb.XGBClassifier(
                random_state=42,        # Reproducibility
                n_jobs=-1,             # Parallel processing
                eval_metric='logloss', # Evaluation metric
                use_label_encoder=False # Avoid deprecation warning
            )
            
            # Perform grid search with cross-validation
            print("   Performing extensive grid search (this may take a while)...")
            grid_search = GridSearchCV(
                xgb_model,
                param_grid,
                cv=self.cv_folds,         # Cross-validation folds
                scoring="roc_auc",        # Optimization metric
                n_jobs=-1,                # Parallel processing
                verbose=1,                # Progress monitoring
            )
            
            # Fit grid search to find best hyperparameters
            grid_search.fit(X_train, y_train)

            # Extract best model with optimal parameters
            best_xgb = grid_search.best_estimator_
            print(f"   Best parameters: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")

            # Generate predictions for evaluation
            y_pred = best_xgb.predict(X_test)
            y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Log parameters and metrics to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Use XGBoost-specific MLflow logging for better integration
            mlflow.xgboost.log_model(best_xgb, "model")

            # Analyze feature importance using XGBoost's built-in importance
            print("   Analyzing XGBoost feature importance...")
            feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": best_xgb.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            # Log top feature importances
            for i, row in feature_importance.head(10).iterrows():
                mlflow.log_metric(
                    f"feature_importance_{row['feature']}", row["importance"]
                )
            
            print("   Top 5 most important features:")
            for i, row in feature_importance.head(5).iterrows():
                print(f"     {row['feature']}: {row['importance']:.4f}")

            # Cross-validation evaluation
            print("   Performing cross-validation...")
            cv_scores = cross_val_score(
                best_xgb, X_train, y_train, 
                cv=self.cv_folds, 
                scoring="roc_auc"
            )
            
            # Log cross-validation statistics
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())

            # Store complete model information
            self.models["xgboost"] = {
                "model": best_xgb,
                "metrics": metrics,
                "cv_scores": cv_scores,
                "feature_importance": feature_importance,
                "best_params": grid_search.best_params_
            }

            print(f"✅ XGBoost training completed!")
            print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return best_xgb, metrics

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate comprehensive evaluation metrics for binary classification.
        
        This private method computes a full suite of classification metrics that
        provide different perspectives on model performance. These metrics are
        essential for understanding model behavior in credit risk applications.
        
        Args:
            y_true (array-like): True binary labels (0 or 1)
            y_pred (array-like): Predicted binary labels (0 or 1)
            y_pred_proba (array-like): Predicted probabilities for positive class
        
        Returns:
            dict: Dictionary containing all calculated metrics:
                - accuracy: Overall prediction accuracy
                - precision: Precision for positive class (creditworthy)
                - recall: Recall for positive class (sensitivity)
                - f1_score: Harmonic mean of precision and recall
                - roc_auc: Area under ROC curve (primary metric)
        
        Metrics Explanation:
            - Accuracy: (TP + TN) / (TP + TN + FP + FN) - Overall correctness
            - Precision: TP / (TP + FP) - Accuracy of positive predictions
            - Recall: TP / (TP + FN) - Ability to find all positive cases
            - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
            - ROC-AUC: Area under Receiver Operating Characteristic curve
        """
        metrics = {
            # Overall accuracy: fraction of correct predictions
            "accuracy": accuracy_score(y_true, y_pred),
            
            # Precision: fraction of positive predictions that are correct
            # Important for credit risk - minimizes false approvals
            "precision": precision_score(y_true, y_pred),
            
            # Recall (Sensitivity): fraction of actual positives correctly identified
            # Important for credit risk - captures creditworthy applicants
            "recall": recall_score(y_true, y_pred),
            
            # F1-Score: harmonic mean of precision and recall
            # Balances both false positives and false negatives
            "f1_score": f1_score(y_true, y_pred),
            
            # ROC-AUC: Area under ROC curve (primary metric)
            # Measures ability to discriminate between classes across all thresholds
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }
        return metrics

    def train_all_models(self):
        """
        Train all supported models and perform comprehensive comparison.
        
        This method orchestrates the complete model training pipeline:
        1. Prepares data using the preprocessing pipeline
        2. Trains all three models (Logistic Regression, Random Forest, XGBoost)
        3. Evaluates and compares their performance
        4. Selects the best model based on ROC-AUC score
        5. Saves the best model for production use
        
        The method provides comprehensive logging and comparison of all models,
        making it easy to understand which algorithm works best for the dataset.
        
        Returns:
            dict: Dictionary containing all trained models and their results
        
        Side Effects:
            - Sets self.best_model to the highest performing model
            - Sets self.best_score to the best ROC-AUC score
            - Saves the best model to disk for production use
            - Logs all experiments to MLflow for tracking
        
        Example:
            >>> trainer = CreditRiskModelTrainer()
            >>> results = trainer.train_all_models()
            >>> print(f"Best model: {trainer.best_model}")
        """
        print("🚀 Starting comprehensive model training pipeline...")
        print("=" * 60)
        
        # Step 1: Prepare data for training
        print("1️⃣ Preparing data for model training...")
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Step 2: Train all models with hyperparameter optimization
        print(f"\n2️⃣ Training multiple models with hyperparameter tuning...")
        
        # Train Logistic Regression
        print(f"\n📈 Training Model 1/3: Logistic Regression")
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Train Random Forest
        print(f"\n🌲 Training Model 2/3: Random Forest")
        self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Train XGBoost
        print(f"\n🚀 Training Model 3/3: XGBoost")
        self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Step 3: Compare models and select best performer
        print(f"\n3️⃣ Comparing model performance...")
        self._select_best_model()
        
        # Step 4: Save best model for production
        print(f"\n4️⃣ Saving best model for production use...")
        self._save_best_model()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return self.models

    def _select_best_model(self):
        """
        Select the best performing model based on ROC-AUC score.
        
        This method compares all trained models using their test set ROC-AUC scores
        and selects the best performer. ROC-AUC is chosen as the primary metric
        because it's robust to class imbalance and measures the model's ability
        to discriminate between creditworthy and non-creditworthy applicants.
        
        Side Effects:
            - Sets self.best_model to the highest performing model
            - Sets self.best_score to the best ROC-AUC score
            - Logs detailed comparison of all models
        
        Comparison Criteria:
            - Primary: ROC-AUC score (discrimination ability)
            - Secondary: Cross-validation stability
            - Tertiary: Model interpretability and complexity
        """
        print("📊 Comparing model performance across all algorithms...")
        
        best_model_name = None
        best_score = 0
        
        # Compare all trained models
        for model_name, model_info in self.models.items():
            current_score = model_info["metrics"]["roc_auc"]
            cv_mean = model_info["cv_scores"].mean()
            cv_std = model_info["cv_scores"].std()
            
            print(f"   {model_name.upper()}:")
            print(f"     Test ROC-AUC: {current_score:.4f}")
            print(f"     CV ROC-AUC: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"     Accuracy: {model_info['metrics']['accuracy']:.4f}")
            print(f"     Precision: {model_info['metrics']['precision']:.4f}")
            print(f"     Recall: {model_info['metrics']['recall']:.4f}")
            print(f"     F1-Score: {model_info['metrics']['f1_score']:.4f}")
            
            # Select model with highest ROC-AUC score
            if current_score > best_score:
                best_score = current_score
                best_model_name = model_name
                self.best_model = model_info["model"]
                self.best_score = best_score
            
            print()  # Empty line for readability
        
        # Log the selection results
        print(f"🏆 BEST MODEL SELECTED: {best_model_name.upper()}")
        print(f"   Best ROC-AUC Score: {best_score:.4f}")
        print(f"   Model Type: {type(self.best_model).__name__}")
        
        # Log additional information about the best model
        if best_model_name in self.models:
            best_model_info = self.models[best_model_name]
            print(f"   Best Parameters: {best_model_info.get('best_params', 'N/A')}")
            
            # Show feature importance if available
            if 'feature_importance' in best_model_info:
                print(f"   Top 3 Important Features:")
                for i, row in best_model_info['feature_importance'].head(3).iterrows():
                    print(f"     {row['feature']}: {row['importance']:.4f}")

    def _save_best_model(self):
        """
        Save the best performing model to disk for production deployment.
        
        This method saves the best model using joblib serialization with a
        timestamped filename for version control. The saved model can be
        loaded later for making predictions on new credit applications.
        
        The model is saved in the 'models' directory with a filename that
        includes the timestamp for easy identification and version management.
        
        Side Effects:
            - Creates 'models' directory if it doesn't exist
            - Saves the best model with timestamp in filename
            - Logs save location and file information
        
        File Format:
            - Filename: best_model_YYYYMMDD_HHMMSS.joblib
            - Location: models/ directory
            - Content: Complete fitted model object
        
        Raises:
            ValueError: If no best model has been selected
            OSError: If unable to create directory or save file
        """
        if self.best_model is None:
            raise ValueError("No best model available. Run train_all_models() first.")
        
        # Create models directory if it doesn't exist
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate timestamped filename for version control
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_model_{timestamp}.joblib"
        filepath = os.path.join(models_dir, filename)
        
        print(f"💾 Saving best model to production...")
        print(f"   Model type: {type(self.best_model).__name__}")
        print(f"   ROC-AUC score: {self.best_score:.4f}")
        print(f"   Save location: {filepath}")
        
        # Save the model using joblib for efficient serialization
        joblib.dump(self.best_model, filepath)
        
        # Verify the save was successful
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"✅ Model saved successfully!")
            print(f"   File size: {file_size:,} bytes")
        else:
            raise OSError(f"Failed to save model to {filepath}")

    def generate_model_report(self):
        """
        Generate comprehensive model training and evaluation report.
        
        This method creates a detailed summary report of the entire model training
        process, including performance metrics, model comparisons, and recommendations.
        The report is useful for documentation, presentations, and decision making.
        
        Returns:
            str: Formatted report string containing:
                - Training summary
                - Model comparison table
                - Best model details
                - Performance metrics
                - Recommendations
        
        Example:
            >>> trainer = CreditRiskModelTrainer()
            >>> trainer.train_all_models()
            >>> report = trainer.generate_model_report()
            >>> print(report)
        """
        if not self.models:
            return "No models have been trained yet. Run train_all_models() first."
        
        # Build comprehensive report
        report = []
        report.append("=" * 80)
        report.append("🎯 GERMAN CREDIT RISK MODEL TRAINING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Training Summary
        report.append("📊 TRAINING SUMMARY")
        report.append("-" * 40)
        report.append(f"Models Trained: {len(self.models)}")
        report.append(f"Features Used: {len(self.feature_names) if self.feature_names else 'Unknown'}")
        report.append(f"Cross-Validation Folds: {self.cv_folds}")
        report.append(f"Primary Metric: ROC-AUC")
        report.append("")
        
        # Model Performance Comparison
        report.append("🏆 MODEL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        report.append(f"{'Model':<15} {'ROC-AUC':<8} {'CV-AUC':<12} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}")
        report.append("-" * 80)
        
        for model_name, model_info in self.models.items():
            metrics = model_info["metrics"]
            cv_mean = model_info["cv_scores"].mean()
            
            # Mark the best model with an asterisk
            marker = " *" if model_info["model"] == self.best_model else "  "
            
            report.append(
                f"{model_name.title():<15} "
                f"{metrics['roc_auc']:.4f}{marker:<6} "
                f"{cv_mean:.4f}±{model_info['cv_scores'].std():.3f} "
                f"{metrics['accuracy']:.4f}   "
                f"{metrics['precision']:.4f}    "
                f"{metrics['recall']:.4f}  "
                f"{metrics['f1_score']:.4f}"
            )
        
        report.append("")
        report.append("* Best performing model")
        report.append("")
        
        # Best Model Details
        if self.best_model is not None:
            best_model_name = None
            for name, info in self.models.items():
                if info["model"] == self.best_model:
                    best_model_name = name
                    break
            
            report.append("🌟 BEST MODEL DETAILS")
            report.append("-" * 40)
            report.append(f"Algorithm: {best_model_name.title()}")
            report.append(f"Model Type: {type(self.best_model).__name__}")
            report.append(f"ROC-AUC Score: {self.best_score:.4f}")
            
            if best_model_name in self.models:
                best_info = self.models[best_model_name]
                report.append(f"Best Parameters: {best_info.get('best_params', 'N/A')}")
                
                # Add feature importance if available
                if 'feature_importance' in best_info:
                    report.append("")
                    report.append("🔍 TOP 10 MOST IMPORTANT FEATURES")
                    report.append("-" * 40)
                    for i, row in best_info['feature_importance'].head(10).iterrows():
                        report.append(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
            
            report.append("")
        
        # Performance Interpretation
        report.append("📈 PERFORMANCE INTERPRETATION")
        report.append("-" * 40)
        if self.best_score >= 0.8:
            performance_level = "Excellent"
        elif self.best_score >= 0.7:
            performance_level = "Good"
        elif self.best_score >= 0.6:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        report.append(f"Overall Performance: {performance_level} (ROC-AUC: {self.best_score:.4f})")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        if self.best_score >= 0.75:
            report.append("✅ Model is ready for production deployment")
            report.append("✅ Performance is suitable for credit risk assessment")
        else:
            report.append("⚠️ Consider additional feature engineering")
            report.append("⚠️ May need more training data or different algorithms")
        
        report.append("✅ Monitor model performance in production")
        report.append("✅ Retrain periodically with new data")
        report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """
    Main function demonstrating complete model training workflow.
    
    This function serves as both a demonstration and a standalone script for
    training credit risk prediction models. It performs the complete workflow:
    1. Model trainer initialization with MLflow tracking
    2. Data preparation and preprocessing
    3. Training multiple algorithms with hyperparameter optimization
    4. Model comparison and selection
    5. Best model saving for production
    6. Comprehensive reporting
    
    The function includes extensive logging and error handling to ensure
    robust execution and easy debugging.
    
    Example Usage:
        python model_training.py
    """
    print("🚀 Starting German Credit Risk Model Training Pipeline")
    print("=" * 60)
    
    try:
        # Initialize the model trainer with MLflow experiment tracking
        print("1️⃣ Initializing Model Trainer...")
        trainer = CreditRiskModelTrainer(experiment_name="german_credit_risk_production")
        
        # Train all models with comprehensive evaluation
        print("\n2️⃣ Training Multiple ML Models...")
        results = trainer.train_all_models()
        
        # Generate and display comprehensive report
        print("\n3️⃣ Generating Model Training Report...")
        report = trainer.generate_model_report()
        print(report)
        
        # Final success message
        print("\n" + "=" * 60)
        print("🎉 MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   Models trained: {len(results)}")
        print(f"   Best model: {type(trainer.best_model).__name__}")
        print(f"   Best ROC-AUC: {trainer.best_score:.4f}")
        print(f"   MLflow experiment: {trainer.experiment_name}")
        print(f"   Model saved: ✅")
        print("\n🚀 Ready for API deployment!")
        
    except Exception as e:
        print(f"\n❌ MODEL TRAINING FAILED!")
        print(f"Error: {str(e)}")
        print("\nPlease check:")
        print("- Data file exists and is accessible")
        print("- All dependencies are installed")
        print("- Sufficient memory and disk space")
        raise


# Execute main function when script is run directly
if __name__ == "__main__":
    main()
