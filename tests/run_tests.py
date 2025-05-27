#!/usr/bin/env python3
"""
Simple test runner for the preprocessing module without pytest dependencies.
"""

import sys
import os
import unittest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import GermanCreditPreprocessor
import pandas as pd
import numpy as np


class TestGermanCreditPreprocessor(unittest.TestCase):
    """Test cases for GermanCreditPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = GermanCreditPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
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
        })
    
    def test_column_names(self):
        """Test that column names are correctly defined."""
        self.assertEqual(len(self.preprocessor.column_names), 21)
        self.assertIn('creditability', self.preprocessor.column_names)
        self.assertIn('checking_account_status', self.preprocessor.column_names)
        self.assertIn('duration_months', self.preprocessor.column_names)
        print("✓ Column names test passed")
    
    def test_feature_lists(self):
        """Test that numerical and categorical features are correctly identified."""
        self.assertEqual(len(self.preprocessor.numerical_features), 7)
        self.assertEqual(len(self.preprocessor.categorical_features), 13)
        self.assertIn('age', self.preprocessor.numerical_features)
        self.assertIn('credit_amount', self.preprocessor.numerical_features)
        self.assertIn('checking_account_status', self.preprocessor.categorical_features)
        self.assertIn('purpose', self.preprocessor.categorical_features)
        print("✓ Feature lists test passed")
    
    def test_preprocessing_pipeline_creation(self):
        """Test that preprocessing pipeline is created correctly."""
        pipeline = self.preprocessor.create_preprocessing_pipeline()
        self.assertIsNotNone(pipeline)
        # Check that transformers are defined (before fitting)
        transformer_names = [name for name, _, _ in pipeline.transformers]
        self.assertIn('num', transformer_names)
        self.assertIn('cat', transformer_names)
        print("✓ Pipeline creation test passed")
    
    def test_fit_transform(self):
        """Test the fit_transform method."""
        X, y, feature_names = self.preprocessor.fit_transform(self.sample_data)
        
        # Check shapes
        self.assertEqual(X.shape[0], 3)  # 3 samples
        self.assertGreater(X.shape[1], 20)  # More features after one-hot encoding
        self.assertEqual(len(y), 3)
        self.assertEqual(len(feature_names), X.shape[1])
        
        # Check target values
        self.assertEqual(list(y), [1, 0, 1])
        
        # Check that preprocessor is fitted
        self.assertIsNotNone(self.preprocessor.preprocessor)
        self.assertIsNotNone(self.preprocessor.feature_names)
        print("✓ Fit transform test passed")
    
    def test_data_split(self):
        """Test data splitting functionality."""
        X, y, _ = self.preprocessor.fit_transform(self.sample_data)
        
        # Test with small dataset (should work even with 3 samples)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            X, y, test_size=0.33, stratify=False
        )
        
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        print("✓ Data split test passed")
    
    def test_transform_new_data(self):
        """Test transforming new data with fitted preprocessor."""
        # Fit on sample data
        X, y, _ = self.preprocessor.fit_transform(self.sample_data)
        
        # Create new data (without target)
        new_data = self.sample_data.drop('creditability', axis=1).iloc[:1]
        
        # Transform new data
        X_new = self.preprocessor.transform(new_data)
        
        self.assertEqual(X_new.shape[0], 1)
        self.assertEqual(X_new.shape[1], X.shape[1])
        print("✓ Transform new data test passed")
    
    def test_invalid_transform_without_fit(self):
        """Test that transform raises error when preprocessor is not fitted."""
        new_data = self.sample_data.drop('creditability', axis=1)
        
        with self.assertRaises(ValueError):
            self.preprocessor.transform(new_data)
        print("✓ Invalid transform test passed")


def run_tests():
    """Run all tests."""
    print("Running German Credit Preprocessor Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGermanCreditPreprocessor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    print("=" * 50)
    if result.wasSuccessful():
        print(f"✅ All {result.testsRun} tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(failure[1])
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(error[1])
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 