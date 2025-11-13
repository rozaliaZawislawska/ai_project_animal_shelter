import pytest
import pandas as pd
import numpy as np
from animalshelter.pipelines.preprocessing.nodes import clean_data, scale_data

TEST_DATA_SIZE = 100
TOLERANCE = 0.05


@pytest.fixture
def sample_raw_data():
    """Creates a minimal raw DataFrame for testing purposes."""
    data = {
        'Animal ID': range(TEST_DATA_SIZE),
        'Type': ['Dog'] * 50 + ['Cat'] * 50,
        'Sex': ['Male'] * 97 + [np.nan] * 3,  # 3 missing values to be dropped
        'Spayed/Neutered': ['Yes'] * 97 + [np.nan] * 3,
        'Primary Breed': ['Labrador'] * 100,
        'Primary Color': ['Black'] * 100,
        'Secondary Color': ['Missing'] * 80 + [np.nan] * 20, # 20 missing values for imputation test
        'Days in Shelter': [float(i) for i in range(TEST_DATA_SIZE)], # Data for scaling
        'Name': ['Buddy'] * 100,
        'Date of Birth': pd.to_datetime(pd.date_range(start='2019-01-01', periods=TEST_DATA_SIZE, freq='7D')),
        'Intake Date': pd.to_datetime(['2021-01-01'] * 100),
        'Outcome Date': pd.to_datetime(['2021-01-10'] * 100),
        'Outcome Status': ['Adopted'] * 100,
        'Euthanasia Reason': [np.nan] * 100,
        'is_adopted': [1] * 100 
    }
    df = pd.DataFrame(data).head(TEST_DATA_SIZE)
    
    # Simulate rows with NaNs in key columns that are dropped by clean_data
    return df

@pytest.fixture
def sample_cleaned_data(sample_raw_data):
    """Returns the DataFrame after the clean_data function."""
    return clean_data(sample_raw_data)

@pytest.fixture
def sample_scaled_data(sample_cleaned_data):
    """Returns the DataFrame after the scale_data function."""
    return scale_data(sample_cleaned_data)


# --- clean_data TESTS ---

def test_clean_data_row_count_and_not_empty(sample_raw_data):
    """Checks if the row count is correct after dropping NaNs and if DataFrame is not empty."""
    df_cleaned = clean_data(sample_raw_data)
    
    # In the raw data, 3 rows have NaNs in 'Sex' or 'Spayed/Neutered', which are dropped.
    expected_rows = len(sample_raw_data) - 3 
    
    # Check 1: DataFrame must not be empty
    assert not df_cleaned.empty, "The cleaned DataFrame is empty."
    
    # Check 2: Row count must match expectation
    assert len(df_cleaned) == expected_rows, \
        f"Expected {expected_rows} rows after dropping NaNs, got {len(df_cleaned)}."

def test_clean_data_no_nan_after_imputation(sample_raw_data):
    df_cleaned = clean_data(sample_raw_data)
    
    assert df_cleaned['Secondary Color'].isnull().sum() == 0, \
        "Column 'Secondary Color' should be fully imputed and contain no NaNs."
    
    assert df_cleaned['age_at_intake'].isnull().sum() == 0, \
        "Column 'age_at_intake' should be fully imputed."



# --- scale_data TESTS ---

def test_scaled_data_is_standardized(sample_scaled_data):
    """Checks if numerical data is standardized (mean ≈ 0, std ≈ 1)."""
    
    scaled_cols = ['Days in Shelter', 'age_at_intake']
    
    for col in scaled_cols:
        mean = sample_scaled_data[col].mean()
        std = sample_scaled_data[col].std()
        
        # Check: Mean ≈ 0
        assert np.isclose(mean, 0.0, atol=TOLERANCE), \
            f"Mean for {col} should be close to 0, is {mean:.3f}."
        
        # Check: Standard Deviation ≈ 1
        assert np.isclose(std, 1.0, atol=TOLERANCE), \
            f"Standard deviation for {col} should be close to 1, is {std:.3f}."

