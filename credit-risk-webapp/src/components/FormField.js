import React from 'react';

const FormField = ({ 
  label, 
  name, 
  type = 'text', 
  value, 
  onChange, 
  options = [], 
  error, 
  required = false,
  min,
  max,
  step,
  placeholder 
}) => {
  const renderInput = () => {
    if (type === 'select') {
      return (
        <select
          name={name}
          value={value}
          onChange={onChange}
          className={`form-input ${error ? 'border-danger-500 focus:border-danger-500 focus:ring-danger-500' : ''}`}
          required={required}
        >
          <option value="">Select an option</option>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      );
    }

    return (
      <input
        type={type}
        name={name}
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        className={`form-input ${error ? 'border-danger-500 focus:border-danger-500 focus:ring-danger-500' : ''}`}
        required={required}
      />
    );
  };

  return (
    <div className="mb-4">
      <label htmlFor={name} className="form-label">
        {label}
        {required && <span className="text-danger-500 ml-1">*</span>}
      </label>
      {renderInput()}
      {error && <p className="form-error">{error}</p>}
    </div>
  );
};

export default FormField; 