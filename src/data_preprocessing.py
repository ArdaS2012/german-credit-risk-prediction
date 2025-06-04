"""
Data preprocessing module for German Credit Dataset.

This module provides comprehensive data preprocessing functionality for the UCI German 
Credit Dataset. It handles data loading, cleaning, categorical encoding, numerical 
scaling, and train-test splitting with a complete pipeline that can be fitted, 
transformed, and persisted for production use.

Key Features:
- Automatic categorical and numerical feature identification
- One-hot encoding for categorical variables
- StandardScaler normalization for numerical features
- Train-test splitting with stratification support
- Serialization/deserialization of fitted preprocessing pipeline
- Feature name tracking through transformations

Author: German Credit Risk Prediction System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


class GermanCreditPreprocessor:
    """
    Complete preprocessing pipeline for German Credit Dataset.
    
    This class handles all data preprocessing steps required to transform raw German 
    Credit Dataset into machine learning ready format. It automatically identifies 
    categorical and numerical features, applies appropriate transformations, and 
    maintains consistency between training and inference.
    
    Attributes:
        column_names (list): Complete list of feature and target column names
        numerical_features (list): Names of numerical features for scaling
        categorical_features (list): Names of categorical features for encoding
        preprocessor (ColumnTransformer): Fitted sklearn preprocessing pipeline
        feature_names (list): Names of features after transformation
    
    Example:
        >>> preprocessor = GermanCreditPreprocessor()
        >>> df = preprocessor.load_data("data/german.data")
        >>> X_processed, y, feature_names = preprocessor.fit_transform(df)
        >>> X_train, X_test, y_train, y_test = preprocessor.split_data(X_processed, y)
    """

    def __init__(self):
        """
        Initialize the German Credit Preprocessor.
        
        Sets up column names, feature categorizations, and empty preprocessing 
        components. The column names correspond to the UCI German Credit Dataset 
        format with proper feature identification.
        """
        # Complete column names for the German Credit Dataset
        # These match the original UCI dataset structure with meaningful names
        self.column_names = [
            "checking_account_status",  # A11, A12, A13, A14 - Checking account status
            "duration_months",  # numerical - Duration of credit in months
            "credit_history",  # A30, A31, A32, A33, A34 - Credit history
            "purpose",  # A40-A410 - Purpose of credit
            "credit_amount",  # numerical - Credit amount in Deutsche Marks
            "savings_account",  # A61-A65 - Savings account/bonds
            "employment_since",  # A71-A75 - Present employment duration
            "installment_rate",  # numerical - Installment rate in % of disposable income
            "personal_status_sex",  # A91-A95 - Personal status and sex
            "other_debtors",  # A101-A103 - Other debtors/guarantors
            "residence_since",  # numerical - Present residence since (years)
            "property",  # A121-A124 - Property ownership
            "age",  # numerical - Age in years
            "other_installment_plans",  # A141-A143 - Other installment plans
            "housing",  # A151-A153 - Housing situation
            "existing_credits",  # numerical - Number of existing credits
            "job",  # A171-A174 - Job category
            "dependents",  # numerical - Number of dependents
            "telephone",  # A191-A192 - Telephone availability
            "foreign_worker",  # A201-A202 - Foreign worker status
            "creditability",  # target: 1=good, 2=bad (will be converted to 1=good, 0=bad)
        ]

        # Numerical features that require scaling
        # These are continuous variables that benefit from standardization
        self.numerical_features = [
            "duration_months",      # 1-72 months
            "credit_amount",        # 250-18424 DM
            "installment_rate",     # 1-4 (percentage)
            "residence_since",      # 1-4 years
            "age",                  # 19-75 years
            "existing_credits",     # 1-4 credits
            "dependents",           # 1-2 people
        ]

        # Categorical features that require encoding
        # These are ordinal and nominal variables with specific code values
        self.categorical_features = [
            "checking_account_status",      # Account balance categories
            "credit_history",               # Payment behavior history
            "purpose",                      # Reason for credit
            "savings_account",              # Savings amount categories
            "employment_since",             # Employment duration categories
            "personal_status_sex",          # Marital status and gender
            "other_debtors",               # Guarantor information
            "property",                     # Property ownership type
            "other_installment_plans",      # Other credit plans
            "housing",                      # Housing situation
            "job",                          # Employment category
            "telephone",                    # Phone availability
            "foreign_worker",               # Citizenship status
        ]

        # Initialize empty preprocessing components
        # These will be populated during fitting
        self.preprocessor = None
        self.feature_names = None

    def load_data(self, data_path="../data/german.data"):
        """
        Load and perform initial processing of the German Credit Dataset.
        
        This method reads the UCI German Credit Dataset from a space-separated file,
        assigns proper column names, and converts the target variable to binary format
        suitable for classification (1=creditworthy, 0=not creditworthy).
        
        Args:
            data_path (str): Path to the german.data file. Defaults to "../data/german.data"
                           The file should be in UCI format with space-separated values.
        
        Returns:
            pd.DataFrame: Loaded dataset with proper column names and processed target.
                         Shape: (1000, 21) - 1000 samples, 20 features + 1 target
        
        Raises:
            FileNotFoundError: If the data file cannot be found at the specified path
            ValueError: If the data format is incorrect or corrupted
        
        Example:
            >>> preprocessor = GermanCreditPreprocessor()
            >>> df = preprocessor.load_data("data/german.data")
            >>> print(f"Dataset shape: {df.shape}")
            Dataset shape: (1000, 21)
        """
        try:
            # Load the dataset from space-separated file without headers
            # The UCI German Credit Dataset comes in this specific format
            df = pd.read_csv(data_path, sep=" ", header=None, names=self.column_names)
            
            # Validate dataset shape
            if df.shape != (1000, 21):
                print(f"Warning: Expected shape (1000, 21), got {df.shape}")
            
            # Convert target variable from original format (1=good, 2=bad) 
            # to binary classification format (1=creditworthy, 0=not creditworthy)
            # This makes the model output more intuitive for business use
            df["creditability"] = df["creditability"].map({1: 1, 2: 0})
            
            # Verify target conversion was successful
            if df["creditability"].isnull().any():
                raise ValueError("Target variable conversion failed - unexpected values found")
            
            # Log dataset information for monitoring and debugging
            print(f"✅ Dataset loaded successfully from {data_path}")
            print(f"📊 Dataset shape: {df.shape}")
            print(f"🎯 Target distribution:")
            print(f"   Creditworthy (1): {(df['creditability'] == 1).sum()} ({(df['creditability'] == 1).mean():.1%})")
            print(f"   Not Creditworthy (0): {(df['creditability'] == 0).sum()} ({(df['creditability'] == 0).mean():.1%})")
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {data_path}. Please ensure the file exists.")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")

    def create_preprocessing_pipeline(self):
        """
        Create comprehensive preprocessing pipeline for numerical and categorical features.
        
        This method constructs a scikit-learn ColumnTransformer that applies appropriate
        preprocessing to different feature types:
        - Numerical features: StandardScaler (z-score normalization)
        - Categorical features: OneHotEncoder (binary encoding)
        
        The pipeline ensures consistent preprocessing between training and inference,
        handles unknown categories gracefully, and maintains feature order.
        
        Returns:
            ColumnTransformer: Configured preprocessing pipeline ready for fitting.
                              Transforms numerical features with StandardScaler and
                              categorical features with OneHotEncoder.
        
        Technical Details:
            - StandardScaler: Transforms features to have mean=0 and std=1
            - OneHotEncoder: Creates binary columns for each category
            - drop='first': Prevents multicollinearity by dropping first category
            - sparse_output=False: Returns dense arrays for compatibility
        
        Example:
            >>> preprocessor = GermanCreditPreprocessor()
            >>> pipeline = preprocessor.create_preprocessing_pipeline()
            >>> print(type(pipeline))
            <class 'sklearn.compose._column_transformer.ColumnTransformer'>
        """
        # Numerical preprocessing pipeline
        # StandardScaler normalizes features to have zero mean and unit variance
        # This is crucial for algorithms sensitive to feature scales (SVM, Neural Networks, etc.)
        numerical_transformer = StandardScaler()
        
        # Categorical preprocessing pipeline
        # OneHotEncoder creates binary indicator variables for each category
        # drop='first' prevents multicollinearity by removing one category per feature
        # sparse_output=False returns dense arrays for easier handling
        categorical_transformer = OneHotEncoder(
            drop="first",           # Prevent multicollinearity
            sparse_output=False,    # Return dense arrays
            handle_unknown="ignore" # Gracefully handle unknown categories during inference
        )

        # Combine preprocessing steps using ColumnTransformer
        # This ensures each feature type gets appropriate preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                # Apply StandardScaler to numerical features
                ("num", numerical_transformer, self.numerical_features),
                # Apply OneHotEncoder to categorical features
                ("cat", categorical_transformer, self.categorical_features),
            ],
            # Keep feature order: numerical features first, then categorical
            remainder="drop",  # Drop any features not specified above
            verbose_feature_names_out=False  # Simplify feature names
        )

        return preprocessor

    def fit_transform(self, df):
        """
        Fit the preprocessing pipeline and transform the data in one step.
        
        This method performs the complete preprocessing workflow:
        1. Separates features from target variable
        2. Creates and fits the preprocessing pipeline on training data
        3. Transforms features using the fitted pipeline
        4. Generates feature names for the transformed data
        5. Provides detailed logging of the transformation process
        
        Args:
            df (pd.DataFrame): Input dataset with features and target variable.
                              Must contain 'creditability' column as target.
                              Shape: (n_samples, 21) where 21 = 20 features + 1 target
        
        Returns:
            tuple: A 3-element tuple containing:
                - X_transformed (np.ndarray): Preprocessed feature matrix
                  Shape: (n_samples, n_transformed_features)
                - y (pd.Series): Target variable (unchanged)
                  Shape: (n_samples,)
                - feature_names (list): Names of transformed features
                  Length: n_transformed_features (typically 48 after one-hot encoding)
        
        Side Effects:
            - Sets self.preprocessor to fitted ColumnTransformer
            - Sets self.feature_names to list of transformed feature names
        
        Raises:
            KeyError: If 'creditability' column is missing from DataFrame
            ValueError: If input data format is invalid
        
        Example:
            >>> preprocessor = GermanCreditPreprocessor()
            >>> df = preprocessor.load_data("data/german.data")
            >>> X_processed, y, feature_names = preprocessor.fit_transform(df)
            >>> print(f"Original features: 20, Transformed features: {len(feature_names)}")
            Original features: 20, Transformed features: 48
        """
        print("🔄 Starting data preprocessing...")
        
        # Validate input data
        if "creditability" not in df.columns:
            raise KeyError("Target column 'creditability' not found in DataFrame")
        
        # Separate features (X) from target variable (y)
        # Features: All columns except the target
        # Target: Binary creditability indicator
        X = df.drop("creditability", axis=1)
        y = df["creditability"]
        
        print(f"📋 Input data validation:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Missing values in features: {X.isnull().sum().sum()}")
        print(f"   Missing values in target: {y.isnull().sum()}")

        # Create and fit the preprocessing pipeline
        # This learns the parameters needed for transformation (means, stds, categories)
        print("⚙️ Creating and fitting preprocessing pipeline...")
        self.preprocessor = self.create_preprocessing_pipeline()
        
        # Transform the features using the fitted pipeline
        # This applies StandardScaler to numerical and OneHotEncoder to categorical features
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Generate comprehensive feature names for the transformed data
        # This is crucial for model interpretability and debugging
        self._get_feature_names()

        # Log transformation results for monitoring and validation
        print("✅ Data preprocessing completed successfully!")
        print(f"📊 Transformation summary:")
        print(f"   Original features: {X.shape[1]}")
        print(f"   Transformed features: {X_transformed.shape[1]}")
        print(f"   Numerical features: {len(self.numerical_features)} -> {len(self.numerical_features)}")
        print(f"   Categorical features: {len(self.categorical_features)} -> {X_transformed.shape[1] - len(self.numerical_features)}")
        print(f"   Feature names generated: {len(self.feature_names)}")
        
        # Validate transformation consistency
        if len(self.feature_names) != X_transformed.shape[1]:
            print("⚠️ Warning: Feature name count doesn't match transformed feature count")

        return X_transformed, y, self.feature_names

    def transform(self, df):
        """
        Transform new data using the previously fitted preprocessing pipeline.
        
        This method applies the already-fitted preprocessing transformations to new data,
        ensuring consistency between training and inference. It's used for making
        predictions on new credit applications using the same preprocessing steps
        that were applied during training.
        
        Args:
            df (pd.DataFrame): Input dataset to transform. Can contain either:
                             - Features only (for inference)
                             - Features + target (target will be ignored)
                             Must have the same feature columns as training data.
        
        Returns:
            np.ndarray: Transformed feature matrix with same structure as training data.
                       Shape: (n_samples, n_transformed_features)
        
        Raises:
            ValueError: If preprocessor hasn't been fitted yet (call fit_transform first)
            KeyError: If required feature columns are missing from input data
        
        Example:
            >>> # After fitting on training data
            >>> X_test_processed = preprocessor.transform(test_df)
            >>> predictions = model.predict(X_test_processed)
        """
        # Validate that preprocessing pipeline has been fitted
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not fitted yet. Call fit_transform() first to train "
                "the preprocessing pipeline on your training data."
            )

        # Handle both inference (features only) and evaluation (features + target) cases
        # Remove target column if present, otherwise use all columns as features
        if "creditability" in df.columns:
            X = df.drop("creditability", axis=1)
            print(f"🔄 Transforming data (with target): {df.shape} -> features: {X.shape}")
        else:
            X = df
            print(f"🔄 Transforming data (inference mode): {X.shape}")
        
        # Validate that all required features are present
        missing_features = set(self.numerical_features + self.categorical_features) - set(X.columns)
        if missing_features:
            raise KeyError(f"Missing required features: {missing_features}")
        
        # Apply the fitted preprocessing pipeline
        # This uses the parameters learned during training (means, stds, categories)
        X_transformed = self.preprocessor.transform(X)
        
        print(f"✅ Transformation completed: {X_transformed.shape}")
        
        return X_transformed

    def _get_feature_names(self):
        """
        Generate comprehensive feature names after preprocessing transformations.
        
        This private method creates human-readable names for all features in the
        transformed dataset. It handles both numerical features (unchanged names)
        and categorical features (with category suffixes after one-hot encoding).
        
        The method supports different sklearn versions and provides fallback logic
        for feature name generation to ensure compatibility.
        
        Side Effects:
            Sets self.feature_names to a list of all transformed feature names
            
        Technical Details:
            - Numerical features keep their original names
            - Categorical features get format: "feature_name_category_value"
            - OneHotEncoder drops first category to prevent multicollinearity
            - Feature order: numerical features first, then categorical features
        
        Example Generated Names:
            Numerical: ['duration_months', 'credit_amount', ...]
            Categorical: ['checking_account_A12', 'checking_account_A13', ...]
        """
        print("🏷️ Generating feature names for transformed data...")
        
        # Start with numerical feature names (unchanged after StandardScaler)
        feature_names_list = []
        
        # Numerical features maintain their original names
        # StandardScaler doesn't change feature names, only scales values
        num_feature_names = self.numerical_features.copy()
        feature_names_list.extend(num_feature_names)
        
        # Generate categorical feature names after one-hot encoding
        # OneHotEncoder creates multiple binary features for each categorical feature
        cat_feature_names = []
        
        # Try modern sklearn method first (v1.0+)
        try:
            if hasattr(self.preprocessor.named_transformers_["cat"], "get_feature_names_out"):
                # Modern sklearn: get_feature_names_out() method
                cat_feature_names = (
                    self.preprocessor.named_transformers_["cat"]
                    .get_feature_names_out(self.categorical_features)
                    .tolist()
                )
                print(f"   Using modern sklearn feature name generation")
            else:
                raise AttributeError("get_feature_names_out not available")
                
        except AttributeError:
            # Fallback for older sklearn versions
            print(f"   Using fallback feature name generation for older sklearn")
            
            # Manually construct feature names from categories
            for i, feature in enumerate(self.categorical_features):
                encoder = self.preprocessor.named_transformers_["cat"]
                if hasattr(encoder, "categories_"):
                    # Get categories for this feature (excluding dropped first category)
                    categories = encoder.categories_[i][1:]  # drop='first' removes first category
                    for cat in categories:
                        # Create feature name: original_feature_category_value
                        cat_feature_names.append(f"{feature}_{cat}")
        
        # Combine numerical and categorical feature names
        feature_names_list.extend(cat_feature_names)
        
        # Store the complete feature name list
        self.feature_names = feature_names_list
        
        # Log feature name generation results
        print(f"✅ Feature name generation completed:")
        print(f"   Numerical feature names: {len(num_feature_names)}")
        print(f"   Categorical feature names: {len(cat_feature_names)}")
        print(f"   Total feature names: {len(self.feature_names)}")
        
        # Show sample feature names for verification
        if len(self.feature_names) > 10:
            print(f"   Sample feature names: {self.feature_names[:5]} ... {self.feature_names[-3:]}")
        else:
            print(f"   All feature names: {self.feature_names}")

    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=True):
        """
        Split preprocessed data into training and testing sets with optional stratification.
        
        This method performs train-test splitting with careful consideration of class
        balance preservation through stratification. It's designed to ensure that
        both training and test sets have representative samples of creditworthy
        and non-creditworthy applications.
        
        Args:
            X (np.ndarray): Preprocessed feature matrix from fit_transform()
                           Shape: (n_samples, n_features)
            y (pd.Series or np.ndarray): Target variable (binary: 0 or 1)
                                       Shape: (n_samples,)
            test_size (float, optional): Proportion of dataset for testing. 
                                       Defaults to 0.2 (20% test, 80% train)
            random_state (int, optional): Random seed for reproducibility. 
                                        Defaults to 42
            stratify (bool, optional): Whether to preserve class distribution.
                                     Defaults to True (recommended for imbalanced data)
        
        Returns:
            tuple: 4-element tuple containing train-test split:
                - X_train (np.ndarray): Training feature matrix
                - X_test (np.ndarray): Testing feature matrix  
                - y_train (np.ndarray): Training target values
                - y_test (np.ndarray): Testing target values
        
        Raises:
            ValueError: If input arrays have mismatched shapes
            ValueError: If test_size is not between 0 and 1
        
        Example:
            >>> X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
            >>> print(f"Training set: {X_train.shape[0]} samples")
            >>> print(f"Test set: {X_test.shape[0]} samples")
            Training set: 800 samples
            Test set: 200 samples
        """
        print("✂️ Splitting data into training and testing sets...")
        
        # Validate input parameters
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: X={len(X)}, y={len(y)}")
        
        # Log split configuration
        print(f"📊 Split configuration:")
        print(f"   Test size: {test_size:.1%}")
        print(f"   Random state: {random_state}")
        print(f"   Stratification: {'Enabled' if stratify else 'Disabled'}")
        
        # Configure stratification parameter
        # Stratify preserves the proportion of samples for each target class
        stratify_param = y if stratify else None
        
        if stratify:
            # Log class distribution before splitting
            unique, counts = np.unique(y, return_counts=True)
            print(f"   Original class distribution:")
            for cls, count in zip(unique, counts):
                print(f"     Class {cls}: {count} samples ({count/len(y):.1%})")
        
        # Perform train-test split using sklearn
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Log splitting results
        print(f"✅ Data splitting completed successfully!")
        print(f"📈 Split results:")
        print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
        print(f"   Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
        print(f"   Feature dimensions: {X_train.shape[1]}")
        
        # Verify stratification worked correctly
        if stratify:
            print(f"   Training class distribution:")
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            for cls, count in zip(unique_train, counts_train):
                print(f"     Class {cls}: {count} samples ({count/len(y_train):.1%})")
            
            print(f"   Test class distribution:")
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            for cls, count in zip(unique_test, counts_test):
                print(f"     Class {cls}: {count} samples ({count/len(y_test):.1%})")

        return X_train, X_test, y_train, y_test

    def save_preprocessor(self, filepath="../models/preprocessor.joblib"):
        """
        Save the fitted preprocessing pipeline to disk for production use.
        
        This method serializes the complete fitted preprocessing pipeline using joblib,
        allowing it to be loaded later for inference on new data. This is essential
        for production deployment where the same preprocessing steps must be applied
        to new credit applications.
        
        Args:
            filepath (str): Path where to save the preprocessor.
                          Defaults to "../models/preprocessor.joblib"
                          Directory will be created if it doesn't exist.
        
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
            OSError: If unable to create directory or write file
        
        Example:
            >>> preprocessor.save_preprocessor("models/preprocessor.joblib")
            ✅ Preprocessor saved successfully to models/preprocessor.joblib
        """
        # Validate that preprocessor has been fitted
        if self.preprocessor is None:
            raise ValueError(
                "Cannot save unfitted preprocessor. Call fit_transform() first."
            )
        
        print(f"💾 Saving fitted preprocessor to {filepath}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the entire preprocessor object including all fitted parameters
        joblib.dump(self, filepath)
        
        print(f"✅ Preprocessor saved successfully!")
        print(f"   File location: {filepath}")
        print(f"   File size: {os.path.getsize(filepath)} bytes")

    def load_preprocessor(self, filepath="../models/preprocessor.joblib"):
        """
        Load a previously saved preprocessing pipeline from disk.
        
        This method deserializes a fitted preprocessing pipeline, restoring all
        learned parameters (means, standard deviations, category encodings) needed
        for consistent preprocessing of new data. Used in production for inference.
        
        Args:
            filepath (str): Path to the saved preprocessor file.
                          Defaults to "../models/preprocessor.joblib"
        
        Raises:
            FileNotFoundError: If the preprocessor file doesn't exist
            ValueError: If the loaded file is corrupted or incompatible
        
        Example:
            >>> new_preprocessor = GermanCreditPreprocessor()
            >>> new_preprocessor.load_preprocessor("models/preprocessor.joblib")
            ✅ Preprocessor loaded successfully from models/preprocessor.joblib
        """
        print(f"📁 Loading preprocessor from {filepath}...")
        
        # Load the saved preprocessor object
        loaded_preprocessor = joblib.load(filepath)
        
        # Copy all attributes from loaded preprocessor to current instance
        self.__dict__.update(loaded_preprocessor.__dict__)
        
        print(f"✅ Preprocessor loaded successfully!")
        print(f"   Features supported: {len(self.feature_names) if self.feature_names else 'Unknown'}")
        print(f"   Fitted: {'Yes' if self.preprocessor else 'No'}")


def main():
    """
    Main function demonstrating complete preprocessing workflow.
    
    This function serves as both a demonstration and a standalone script for
    preprocessing the German Credit Dataset. It performs the complete workflow:
    1. Data loading and validation
    2. Preprocessing pipeline fitting and transformation
    3. Train-test splitting with stratification
    4. Saving fitted components for production use
    
    The function includes comprehensive logging and error handling to ensure
    robust execution and easy debugging.
    
    Example Usage:
        python data_preprocessing.py
    """
    print("🚀 Starting German Credit Dataset Preprocessing Pipeline")
    print("=" * 60)
    
    try:
        # Initialize the preprocessor
        print("1️⃣ Initializing German Credit Preprocessor...")
        preprocessor = GermanCreditPreprocessor()
        
        # Load the dataset
        print("\n2️⃣ Loading German Credit Dataset...")
        df = preprocessor.load_data()
        
        # Perform preprocessing
        print("\n3️⃣ Applying preprocessing transformations...")
        X_transformed, y, feature_names = preprocessor.fit_transform(df)
        
        # Split the data
        print("\n4️⃣ Splitting data for training and testing...")
        X_train, X_test, y_train, y_test = preprocessor.split_data(X_transformed, y)
        
        # Save the fitted preprocessor
        print("\n5️⃣ Saving fitted preprocessor for production use...")
        preprocessor.save_preprocessor()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Final Summary:")
        print(f"   Original dataset: {df.shape}")
        print(f"   Processed features: {X_transformed.shape[1]}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Feature names: {len(feature_names)}")
        print(f"   Preprocessor saved: ✅")
        print("\n🚀 Ready for model training!")
        
    except Exception as e:
        print(f"\n❌ PREPROCESSING FAILED!")
        print(f"Error: {str(e)}")
        raise


# Execute main function when script is run directly
if __name__ == "__main__":
    main()
