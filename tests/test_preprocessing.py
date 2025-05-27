"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import GermanCreditPreprocessor


class TestGermanCreditPreprocessor:
    """Test cases for GermanCreditPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing."""
        return GermanCreditPreprocessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            'checking_account_status': ['A11', 'A12', 'A13'],
            'duration_months': [6, 12, 24],
            'credit_history': ['A30', 'A31', 'A32'],
            'purpose': ['A40', 'A41', 'A42'],
            'credit_amount': [1000, 2000, 3000],
            'savings_account': ['A61', 'A62', 'A63'],
            'employment_since': ['A71', 'A72', 'A73'],
            'installment_rate': [1, 2, 3],
            'personal_status_sex': ['A91', 'A92', 'A93'],
            'other_debtors': ['A101', 'A102', 'A103'],
            'residence_since': [1, 2, 3],
            'property': ['A121', 'A122', 'A123'],
            'age': [25, 35, 45],
            'other_installment_plans': ['A141', 'A142', 'A143'],
            'housing': ['A151', 'A152', 'A153'],
            'existing_credits': [1, 2, 1],
            'job': ['A171', 'A172', 'A173'],
            'dependents': [1, 2, 1],
            'telephone': ['A191', 'A192', 'A191'],
            'foreign_worker': ['A201', 'A202', 'A201'],
            'creditability': [1, 0, 1]
        }
        return pd.DataFrame(data)
    
    def test_column_names(self, preprocessor):
        """Test that column names are correctly defined."""
        assert len(preprocessor.column_names) == 21
        assert 'creditability' in preprocessor.column_names
        assert 'checking_account_status' in preprocessor.column_names
        assert 'duration_months' in preprocessor.column_names
    
    def test_feature_lists(self, preprocessor):
        """Test that numerical and categorical features are correctly identified."""
        assert len(preprocessor.numerical_features) == 7
        assert len(preprocessor.categorical_features) == 13
        assert 'age' in preprocessor.numerical_features
        assert 'credit_amount' in preprocessor.numerical_features
        assert 'checking_account_status' in preprocessor.categorical_features
        assert 'purpose' in preprocessor.categorical_features
    
    def test_preprocessing_pipeline_creation(self, preprocessor):
        """Test that preprocessing pipeline is created correctly."""
        pipeline = preprocessor.create_preprocessing_pipeline()
        assert pipeline is not None
        # Check that transformers are defined (before fitting)
        transformer_names = [name for name, _, _ in pipeline.transformers]
        assert 'num' in transformer_names
        assert 'cat' in transformer_names
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test the fit_transform method."""
        X, y, feature_names = preprocessor.fit_transform(sample_data)
        
        # Check shapes
        assert X.shape[0] == 3  # 3 samples
        assert X.shape[1] > 20  # More features after one-hot encoding
        assert len(y) == 3
        assert len(feature_names) == X.shape[1]
        
        # Check target values
        assert list(y) == [1, 0, 1]
        
        # Check that preprocessor is fitted
        assert preprocessor.preprocessor is not None
        assert preprocessor.feature_names is not None
    
    def test_data_split(self, preprocessor, sample_data):
        """Test data splitting functionality."""
        X, y, _ = preprocessor.fit_transform(sample_data)
        
        # Test with small dataset (should work even with 3 samples)
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X, y, test_size=0.33, stratify=False
        )
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert X_train.shape[1] == X_test.shape[1]
    
    def test_transform_new_data(self, preprocessor, sample_data):
        """Test transforming new data with fitted preprocessor."""
        # Fit on sample data
        X, y, _ = preprocessor.fit_transform(sample_data)
        
        # Create new data (without target)
        new_data = sample_data.drop('creditability', axis=1).iloc[:1]
        
        # Transform new data
        X_new = preprocessor.transform(new_data)
        
        assert X_new.shape[0] == 1
        assert X_new.shape[1] == X.shape[1]
    
    def test_invalid_transform_without_fit(self, preprocessor, sample_data):
        """Test that transform raises error when preprocessor is not fitted."""
        new_data = sample_data.drop('creditability', axis=1)
        
        with pytest.raises(ValueError, match="Preprocessor not fitted"):
            preprocessor.transform(new_data)


class TestDataValidation:
    """Test data validation and edge cases."""
    
    def test_missing_columns(self):
        """Test handling of missing columns."""
        preprocessor = GermanCreditPreprocessor()
        
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'checking_account_status': ['A11'],
            'duration_months': [12],
            'creditability': [1]
        })
        
        # Should raise an error due to missing columns
        with pytest.raises(Exception):
            preprocessor.fit_transform(incomplete_data)
    
    def test_invalid_categorical_values(self):
        """Test handling of invalid categorical values."""
        preprocessor = GermanCreditPreprocessor()
        
        # Create data with invalid categorical value
        data = {
            'checking_account_status': ['INVALID'],  # Invalid value
            'duration_months': [12],
            'credit_history': ['A30'],
            'purpose': ['A40'],
            'credit_amount': [1000],
            'savings_account': ['A61'],
            'employment_since': ['A71'],
            'installment_rate': [1],
            'personal_status_sex': ['A91'],
            'other_debtors': ['A101'],
            'residence_since': [1],
            'property': ['A121'],
            'age': [25],
            'other_installment_plans': ['A141'],
            'housing': ['A151'],
            'existing_credits': [1],
            'job': ['A171'],
            'dependents': [1],
            'telephone': ['A191'],
            'foreign_worker': ['A201'],
            'creditability': [1]
        }
        invalid_data = pd.DataFrame(data)
        
        # Should handle gracefully (one-hot encoder will create new category)
        X, y, _ = preprocessor.fit_transform(invalid_data)
        assert X.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__]) 