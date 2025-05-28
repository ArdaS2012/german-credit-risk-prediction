// Mapping from user-friendly form values to API codes
export const fieldMappings = {
  checkingAccountStatus: {
    '< 0 DM': 'A11',
    '0 ≤ … < 200 DM': 'A12',
    '≥ 200 DM': 'A13',
    'no account': 'A14'
  },
  
  creditHistory: {
    'no credits/all paid': 'A30',
    'all paid': 'A31',
    'existing paid': 'A32',
    'delayed previously': 'A33',
    'critical account': 'A34'
  },
  
  purpose: {
    'car (new)': 'A40',
    'car (used)': 'A41',
    'furniture/equipment': 'A42',
    'radio/TV': 'A43',
    'domestic appliance': 'A44',
    'repairs': 'A45',
    'education': 'A46',
    'retraining': 'A47',
    'business': 'A48',
    'other': 'A49'
  },
  
  savingsAccount: {
    '< 100 DM': 'A61',
    '100 ≤ … < 500 DM': 'A62',
    '500 ≤ … < 1000 DM': 'A63',
    '≥ 1000 DM': 'A64',
    'unknown': 'A65'
  },
  
  employmentSince: {
    'unemployed': 'A71',
    '< 1 year': 'A72',
    '1 ≤ … < 4 years': 'A73',
    '4 ≤ … < 7 years': 'A74',
    '≥ 7 years': 'A75'
  },
  
  personalStatusSex: {
    'male-divorced/separated': 'A91',
    'male-single': 'A92',
    'male-married/widowed': 'A93',
    'female-divorced/separated/married': 'A94'
  },
  
  otherDebtors: {
    'none': 'A101',
    'co-applicant': 'A102',
    'guarantor': 'A103'
  },
  
  property: {
    'real estate': 'A121',
    'building society savings/life insurance': 'A122',
    'car or other': 'A123',
    'unknown / no property': 'A124'
  },
  
  otherInstallmentPlans: {
    'bank': 'A141',
    'stores': 'A142',
    'none': 'A143'
  },
  
  housing: {
    'own': 'A151',
    'rent': 'A152',
    'for free': 'A153'
  },
  
  job: {
    'unemployed/unskilled – non-resident': 'A171',
    'unskilled – resident': 'A172',
    'skilled employee/official': 'A173',
    'highly qualified/self-employed': 'A174'
  },
  
  telephone: {
    'none': 'A191',
    'yes (under customer name)': 'A192'
  },
  
  foreignWorker: {
    'yes': 'A201',
    'no': 'A202'
  }
};

// Convert form data to API format
export const convertFormDataToAPI = (formData) => {
  return {
    checking_account_status: fieldMappings.checkingAccountStatus[formData.checkingAccountStatus],
    duration_months: parseInt(formData.durationMonths),
    credit_history: fieldMappings.creditHistory[formData.creditHistory],
    purpose: fieldMappings.purpose[formData.purpose],
    credit_amount: parseFloat(formData.creditAmount),
    savings_account: fieldMappings.savingsAccount[formData.savingsAccount],
    employment_since: fieldMappings.employmentSince[formData.employmentSince],
    installment_rate: parseInt(formData.installmentRate),
    personal_status_sex: fieldMappings.personalStatusSex[formData.personalStatusSex],
    other_debtors: fieldMappings.otherDebtors[formData.otherDebtors],
    residence_since: parseInt(formData.residenceSince),
    property: fieldMappings.property[formData.property],
    age: parseInt(formData.age),
    other_installment_plans: fieldMappings.otherInstallmentPlans[formData.otherInstallmentPlans],
    housing: fieldMappings.housing[formData.housing],
    existing_credits: parseInt(formData.existingCredits),
    job: fieldMappings.job[formData.job],
    dependents: parseInt(formData.dependents),
    telephone: fieldMappings.telephone[formData.telephone],
    foreign_worker: fieldMappings.foreignWorker[formData.foreignWorker]
  };
}; 