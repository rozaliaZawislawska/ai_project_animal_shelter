import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

def clean_data(processed_data: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = processed_data.copy()
    logger.debug("Missing values before cleaning:")
    logger.debug(df_cleaned.isnull().sum().sort_values(ascending=False).head().to_string())

    # --- CONVERT TYPE OF COLUMN ---

    rows_before = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=['Sex', 'Spayed/Neutered', 'Days in Shelter'])
    rows_after = len(df_cleaned)
    logger.info(f"PREPROCESSING: Dropped {rows_before - rows_after} rows with marginal missing values (Sex/Spayed/Neutered).")

    logger.info("Converting date columns to datetime objects.")
    date_cols = ['Outcome Date', 'Date of Birth', 'Intake Date']
    for col in date_cols:
        df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce', format='mixed')

    # --- IMPUTATION AND FEATURE ENGINEERING ---

    df_cleaned['Secondary Color'] = df_cleaned['Secondary Color'].fillna('Missing')
    logger.info("Imputed Secondary Color with 'Missing'.")

    df_cleaned['Has Name'] = np.where(df_cleaned['Name'].isnull(),0,1)

    df_cleaned['age_at_intake'] = (df_cleaned['Intake Date']- df_cleaned['Date of Birth']).dt.days

    df_cleaned['intake_month'] = df_cleaned['Intake Date'].dt.month
    df_cleaned['intake_dayofweek'] = df_cleaned['Intake Date'].dt.dayofweek

    median_age_days = df_cleaned['age_at_intake'].median()
    df_cleaned['age_at_intake'] = df_cleaned['age_at_intake'].fillna(median_age_days)
    logger.info(f"PREPROCESSING: Created 'age_at_intake' feature and imputed missing values with median: {median_age_days:.0f} days.")

    # --- ELIMINATION AND ROW DROPPING ---

    cols_to_drop = [
        'Animal ID',
        'Outcome Status',
        'Euthanasia Reason',
        'Name',
        'Date of Birth', 
        'Intake Date', 
        'Outcome Date'
    ]
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    logger.info(f"PREPROCESSING: Dropped columns: {cols_to_drop}")

    logger.info(f"Final shape of the DataFrame: {df_cleaned.shape}")
    logger.debug("Missing values after cleaning:")
    logger.debug(df_cleaned.isnull().sum().sort_values(ascending=False).head().to_string())


    return df_cleaned

def scale_data(clean_data: pd.DataFrame) -> pd.DataFrame:
    df_transformed = clean_data.copy()

    categorical_features = [
        'Type', 'Sex', 'Spayed/Neutered', 
        'Primary Breed', 'Primary Color', 'Secondary Color', 
        'intake_month', 'intake_dayofweek' 
    ]
    numerical_features = [
        'Days in Shelter', 'age_at_intake'
    ]
    passthrough_features = ['Has Name', 'is_adopted']

    for col in categorical_features:
        df_transformed[col] = df_transformed[col].astype(str)
    
    # --- CREATE TRANSFORMATION ---

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    scale_data = preprocessor.fit_transform(df_transformed)

    logger.info(f"TRANSFORMATION: Data transformation successful. Final columns count: {scale_data.shape[1]}")
    logger.info(f"TRANSFORMATION: Transformed columns sample: {scale_data.columns[:10].tolist()}")


    return scale_data

def split_data(df_scaled: pd.DataFrame) -> tuple:
    logger.info("Starting data split (70/15/15).")

    X = df_scaled.drop(columns=['is_adopted'])
    y = df_scaled['is_adopted']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42, 
        shuffle=True, 
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=y_temp
    )

    # train (70%)
    train_df = pd.concat([X_train, y_train], axis=1)
    # valid (15%)
    val_df = pd.concat([X_val, y_val], axis=1)
    # test (15%)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    logger.info(f"Train size: {len(train_df)} rows (70%)")
    logger.info(f"Validation size: {len(val_df)} rows (15%)")
    logger.info(f"Test size: {len(test_df)} rows (15%)")

    return train_df, val_df, test_df
