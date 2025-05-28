import React, { useState } from 'react';
import axios from 'axios';
import FormField from './FormField';
import LoadingSpinner from './LoadingSpinner';
import ResultDisplay from './ResultDisplay';
import { fieldMappings, convertFormDataToAPI } from '../utils/fieldMappings';

const CreditRiskForm = () => {
  const [formData, setFormData] = useState({
    checkingAccountStatus: '',
    durationMonths: '',
    creditHistory: '',
    purpose: '',
    creditAmount: '',
    savingsAccount: '',
    employmentSince: '',
    installmentRate: '',
    personalStatusSex: '',
    otherDebtors: '',
    residenceSince: '',
    property: '',
    age: '',
    otherInstallmentPlans: '',
    housing: '',
    existingCredits: '',
    job: '',
    dependents: '',
    telephone: '',
    foreignWorker: ''
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState('');

  const validateForm = () => {
    const newErrors = {};

    // Required field validation
    const requiredFields = [
      'checkingAccountStatus', 'durationMonths', 'creditHistory', 'purpose',
      'creditAmount', 'savingsAccount', 'employmentSince', 'installmentRate',
      'personalStatusSex', 'otherDebtors', 'residenceSince', 'property',
      'age', 'otherInstallmentPlans', 'housing', 'existingCredits',
      'job', 'dependents', 'telephone', 'foreignWorker'
    ];

    requiredFields.forEach(field => {
      if (!formData[field]) {
        newErrors[field] = 'This field is required';
      }
    });

    // Numeric field validation
    if (formData.durationMonths && (formData.durationMonths < 1 || formData.durationMonths > 72)) {
      newErrors.durationMonths = 'Duration must be between 1 and 72 months';
    }

    if (formData.creditAmount && (formData.creditAmount < 250 || formData.creditAmount > 20000)) {
      newErrors.creditAmount = 'Credit amount must be between 250 and 20,000 DM';
    }

    if (formData.installmentRate && (formData.installmentRate < 1 || formData.installmentRate > 4)) {
      newErrors.installmentRate = 'Installment rate must be between 1 and 4';
    }

    if (formData.residenceSince && (formData.residenceSince < 1 || formData.residenceSince > 4)) {
      newErrors.residenceSince = 'Residence since must be between 1 and 4 years';
    }

    if (formData.age && (formData.age < 18 || formData.age > 100)) {
      newErrors.age = 'Age must be between 18 and 100 years';
    }

    if (formData.existingCredits && (formData.existingCredits < 1 || formData.existingCredits > 4)) {
      newErrors.existingCredits = 'Existing credits must be between 1 and 4';
    }

    if (formData.dependents && (formData.dependents < 1 || formData.dependents > 2)) {
      newErrors.dependents = 'Dependents must be 1 or 2';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setApiError('');

    try {
      const apiData = convertFormDataToAPI(formData);
      
      // Note: Using localhost:8000 as specified in requirements
      // In production, this should be an environment variable
      const response = await axios.post('http://localhost:8000/predict', apiData, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 10000, // 10 second timeout
      });

      setResult(response.data);
    } catch (error) {
      console.error('API Error:', error);
      
      if (error.code === 'ECONNABORTED') {
        setApiError('Request timeout. Please try again.');
      } else if (error.response) {
        setApiError(`Server error: ${error.response.status}. Please check if the API is running.`);
      } else if (error.request) {
        setApiError('Unable to connect to the API. Please ensure the API server is running on localhost:8000.');
      } else {
        setApiError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setFormData({
      checkingAccountStatus: '',
      durationMonths: '',
      creditHistory: '',
      purpose: '',
      creditAmount: '',
      savingsAccount: '',
      employmentSince: '',
      installmentRate: '',
      personalStatusSex: '',
      otherDebtors: '',
      residenceSince: '',
      property: '',
      age: '',
      otherInstallmentPlans: '',
      housing: '',
      existingCredits: '',
      job: '',
      dependents: '',
      telephone: '',
      foreignWorker: ''
    });
    setErrors({});
    setApiError('');
  };

  if (result) {
    return <ResultDisplay result={result} onReset={handleReset} />;
  }

  return (
    <div className="card">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Credit Risk Assessment
        </h2>
        <p className="text-gray-600">
          Please fill out all fields to assess your credit risk using our machine learning model.
        </p>
      </div>

      {apiError && (
        <div className="bg-danger-50 border border-danger-200 rounded-md p-4 mb-6">
          <div className="flex">
            <svg className="w-5 h-5 text-danger-400 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div>
              <h3 className="text-sm font-medium text-danger-800">Error</h3>
              <p className="text-sm text-danger-700 mt-1">{apiError}</p>
            </div>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FormField
            label="Status of existing checking account"
            name="checkingAccountStatus"
            type="select"
            value={formData.checkingAccountStatus}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.checkingAccountStatus)}
            error={errors.checkingAccountStatus}
            required
          />

          <FormField
            label="Duration in months"
            name="durationMonths"
            type="number"
            value={formData.durationMonths}
            onChange={handleInputChange}
            min="1"
            max="72"
            error={errors.durationMonths}
            required
          />

          <FormField
            label="Credit history"
            name="creditHistory"
            type="select"
            value={formData.creditHistory}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.creditHistory)}
            error={errors.creditHistory}
            required
          />

          <FormField
            label="Purpose"
            name="purpose"
            type="select"
            value={formData.purpose}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.purpose)}
            error={errors.purpose}
            required
          />

          <FormField
            label="Credit amount (DM)"
            name="creditAmount"
            type="number"
            value={formData.creditAmount}
            onChange={handleInputChange}
            min="250"
            max="20000"
            step="0.01"
            error={errors.creditAmount}
            required
          />

          <FormField
            label="Savings account/bonds"
            name="savingsAccount"
            type="select"
            value={formData.savingsAccount}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.savingsAccount)}
            error={errors.savingsAccount}
            required
          />

          <FormField
            label="Present employment since"
            name="employmentSince"
            type="select"
            value={formData.employmentSince}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.employmentSince)}
            error={errors.employmentSince}
            required
          />

          <FormField
            label="Installment rate (% of disposable income)"
            name="installmentRate"
            type="number"
            value={formData.installmentRate}
            onChange={handleInputChange}
            min="1"
            max="4"
            error={errors.installmentRate}
            required
          />

          <FormField
            label="Personal status and sex"
            name="personalStatusSex"
            type="select"
            value={formData.personalStatusSex}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.personalStatusSex)}
            error={errors.personalStatusSex}
            required
          />

          <FormField
            label="Other debtors/guarantors"
            name="otherDebtors"
            type="select"
            value={formData.otherDebtors}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.otherDebtors)}
            error={errors.otherDebtors}
            required
          />

          <FormField
            label="Present residence since (years)"
            name="residenceSince"
            type="number"
            value={formData.residenceSince}
            onChange={handleInputChange}
            min="1"
            max="4"
            error={errors.residenceSince}
            required
          />

          <FormField
            label="Property"
            name="property"
            type="select"
            value={formData.property}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.property)}
            error={errors.property}
            required
          />

          <FormField
            label="Age in years"
            name="age"
            type="number"
            value={formData.age}
            onChange={handleInputChange}
            min="18"
            max="100"
            error={errors.age}
            required
          />

          <FormField
            label="Other installment plans"
            name="otherInstallmentPlans"
            type="select"
            value={formData.otherInstallmentPlans}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.otherInstallmentPlans)}
            error={errors.otherInstallmentPlans}
            required
          />

          <FormField
            label="Housing"
            name="housing"
            type="select"
            value={formData.housing}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.housing)}
            error={errors.housing}
            required
          />

          <FormField
            label="Number of existing credits"
            name="existingCredits"
            type="number"
            value={formData.existingCredits}
            onChange={handleInputChange}
            min="1"
            max="4"
            error={errors.existingCredits}
            required
          />

          <FormField
            label="Job"
            name="job"
            type="select"
            value={formData.job}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.job)}
            error={errors.job}
            required
          />

          <FormField
            label="Number of people being liable to provide maintenance"
            name="dependents"
            type="number"
            value={formData.dependents}
            onChange={handleInputChange}
            min="1"
            max="2"
            error={errors.dependents}
            required
          />

          <FormField
            label="Telephone"
            name="telephone"
            type="select"
            value={formData.telephone}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.telephone)}
            error={errors.telephone}
            required
          />

          <FormField
            label="Foreign worker"
            name="foreignWorker"
            type="select"
            value={formData.foreignWorker}
            onChange={handleInputChange}
            options={Object.keys(fieldMappings.foreignWorker)}
            error={errors.foreignWorker}
            required
          />
        </div>

        <div className="flex justify-center pt-6">
          <button
            type="submit"
            disabled={loading}
            className="btn-primary min-w-[200px]"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <LoadingSpinner size="sm" />
                <span className="ml-2">Assessing Risk...</span>
              </div>
            ) : (
              'Assess Credit Risk'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default CreditRiskForm; 