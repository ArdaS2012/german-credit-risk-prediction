import React from 'react';

const ResultDisplay = ({ result, onReset }) => {
  const { creditworthy, probability, risk_score } = result;
  
  const riskLevel = creditworthy ? 'Low Risk' : 'High Risk';
  const riskColor = creditworthy ? 'success' : 'danger';
  const confidenceScore = (probability * 100).toFixed(1);
  
  return (
    <div className="card mt-6">
      <div className="text-center">
        <div className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-semibold mb-4 ${
          creditworthy 
            ? 'bg-success-100 text-success-700' 
            : 'bg-danger-100 text-danger-700'
        }`}>
          {creditworthy ? (
            <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          )}
          {riskLevel}
        </div>
        
        <h3 className="text-2xl font-bold text-gray-900 mb-4">
          Credit Risk Assessment Complete
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">
              Creditworthiness
            </h4>
            <p className="text-2xl font-bold text-gray-900">
              {creditworthy ? 'Approved' : 'Declined'}
            </p>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-500 uppercase tracking-wide mb-2">
              Confidence Score
            </h4>
            <p className="text-2xl font-bold text-gray-900">
              {confidenceScore}%
            </p>
          </div>
        </div>
        
        <div className="bg-blue-50 rounded-lg p-4 mb-6">
          <h4 className="text-sm font-medium text-blue-800 mb-2">
            Risk Score Details
          </h4>
          <div className="flex items-center justify-between">
            <span className="text-sm text-blue-700">Risk Score:</span>
            <span className="font-semibold text-blue-900">
              {(risk_score * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-blue-200 rounded-full h-2 mt-2">
            <div 
              className={`h-2 rounded-full ${creditworthy ? 'bg-success-500' : 'bg-danger-500'}`}
              style={{ width: `${risk_score * 100}%` }}
            ></div>
          </div>
        </div>
        
        <div className="text-sm text-gray-600 mb-6">
          <p>
            This assessment is based on the German Credit Dataset and machine learning algorithms. 
            The result should be used as a reference only and not as the sole basis for credit decisions.
          </p>
        </div>
        
        <button
          onClick={onReset}
          className="btn-primary"
        >
          Assess Another Application
        </button>
      </div>
    </div>
  );
};

export default ResultDisplay; 