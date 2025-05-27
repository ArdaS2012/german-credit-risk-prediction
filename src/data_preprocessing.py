"""
Data preprocessing module for German Credit Dataset.
Handles data loading, cleaning, encoding, and feature engineering.
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
    Preprocessor for German Credit Dataset.
    Handles categorical encoding, numerical scaling, and data splitting.
    """

    def __init__(self):
        self.column_names = [
            "checking_account_status",  # A11, A12, A13, A14
            "duration_months",  # numerical
            "credit_history",  # A30, A31, A32, A33, A34
            "purpose",  # A40-A410
            "credit_amount",  # numerical
            "savings_account",  # A61-A65
            "employment_since",  # A71-A75
            "installment_rate",  # numerical
            "personal_status_sex",  # A91-A95
            "other_debtors",  # A101-A103
            "residence_since",  # numerical
            "property",  # A121-A124
            "age",  # numerical
            "other_installment_plans",  # A141-A143
            "housing",  # A151-A153
            "existing_credits",  # numerical
            "job",  # A171-A174
            "dependents",  # numerical
            "telephone",  # A191-A192
            "foreign_worker",  # A201-A202
            "creditability",  # target: 1=good, 2=bad
        ]

        self.numerical_features = [
            "duration_months",
            "credit_amount",
            "installment_rate",
            "residence_since",
            "age",
            "existing_credits",
            "dependents",
        ]

        self.categorical_features = [
            "checking_account_status",
            "credit_history",
            "purpose",
            "savings_account",
            "employment_since",
            "personal_status_sex",
            "other_debtors",
            "property",
            "other_installment_plans",
            "housing",
            "job",
            "telephone",
            "foreign_worker",
        ]

        self.preprocessor = None
        self.feature_names = None

    def load_data(self, data_path="../data/german.data"):
        """
        Load the German Credit Dataset.

        Args:
            data_path (str): Path to the german.data file

        Returns:
            pd.DataFrame: Loaded and initially processed dataset
        """
        # Load the dataset
        df = pd.read_csv(data_path, sep=" ", header=None, names=self.column_names)

        # Convert target variable: 1=creditworthy, 0=not creditworthy (original: 1=good, 2=bad)
        df["creditability"] = df["creditability"].map({1: 1, 2: 0})

        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(f"Target distribution:\n{df['creditability'].value_counts()}")

        return df

    def create_preprocessing_pipeline(self):
        """
        Create preprocessing pipeline for numerical and categorical features.

        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        # Numerical preprocessing: scaling
        numerical_transformer = StandardScaler()

        # Categorical preprocessing: one-hot encoding
        categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        return preprocessor

    def fit_transform(self, df):
        """
        Fit the preprocessing pipeline and transform the data.

        Args:
            df (pd.DataFrame): Input dataset

        Returns:
            tuple: (X_transformed, y, feature_names)
        """
        # Separate features and target
        X = df.drop("creditability", axis=1)
        y = df["creditability"]

        # Create and fit preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline()
        X_transformed = self.preprocessor.fit_transform(X)

        # Get feature names after transformation
        self._get_feature_names()

        print(f"Data preprocessed successfully.")
        print(f"Original features: {X.shape[1]}")
        print(f"Transformed features: {X_transformed.shape[1]}")

        return X_transformed, y, self.feature_names

    def transform(self, df):
        """
        Transform new data using fitted preprocessing pipeline.

        Args:
            df (pd.DataFrame): Input dataset

        Returns:
            np.ndarray: Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        X = df.drop("creditability", axis=1) if "creditability" in df.columns else df
        return self.preprocessor.transform(X)

    def _get_feature_names(self):
        """Get feature names after preprocessing."""
        # Numerical feature names (unchanged)
        num_feature_names = self.numerical_features

        # Categorical feature names (after one-hot encoding)
        cat_feature_names = []
        if hasattr(
            self.preprocessor.named_transformers_["cat"], "get_feature_names_out"
        ):
            cat_feature_names = (
                self.preprocessor.named_transformers_["cat"]
                .get_feature_names_out(self.categorical_features)
                .tolist()
            )
        else:
            # Fallback for older sklearn versions
            for i, feature in enumerate(self.categorical_features):
                encoder = self.preprocessor.named_transformers_["cat"]
                if hasattr(encoder, "categories_"):
                    categories = encoder.categories_[i][1:]  # drop first category
                    for cat in categories:
                        cat_feature_names.append(f"{feature}_{cat}")

        self.feature_names = num_feature_names + cat_feature_names

    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=True):
        """
        Split data into training and testing sets.

        Args:
            X (np.ndarray): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed
            stratify (bool): Whether to stratify split

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
        )

        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")
        print(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")

        return X_train, X_test, y_train, y_test

    def save_preprocessor(self, filepath="../models/preprocessor.joblib"):
        """Save the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
        print(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath="../models/preprocessor.joblib"):
        """Load a fitted preprocessor."""
        self.preprocessor = joblib.load(filepath)
        self._get_feature_names()
        print(f"Preprocessor loaded from {filepath}")


def main():
    """Main function to demonstrate preprocessing pipeline."""
    # Initialize preprocessor
    preprocessor = GermanCreditPreprocessor()

    # Load data
    df = preprocessor.load_data()

    # Preprocess data
    X, y, feature_names = preprocessor.fit_transform(df)

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    # Save preprocessor
    preprocessor.save_preprocessor()

    print(f"\nPreprocessing completed successfully!")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10 features

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    main()
