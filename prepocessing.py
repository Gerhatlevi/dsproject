import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def converting_excel_to_csv(excel_file, csv_file):
    df = pd.read_excel(excel_file)
    df.to_csv(csv_file, index=False)

#converting_excel_to_csv('time_series_375_preprocess_en-2.xlsx', 'train_data_375.csv')
#converting_excel_to_csv('time_series_test_110_preprocess_en-2.xlsx', 'test_data_110.csv')


###Preprocessing test data
def time_weighted_average(group):
    re_date = pd.to_datetime(group['RE_DATE'])
    last_date = re_date.max()
    
    delta_days = (last_date - re_date).dt.days
    
    weights = 0.5 ** (delta_days / 3) # 3-day half-life for weighting, because patients' conditions can change rapidly
    exclude = ['PATIENT_ID', 'RE_DATE', 'outcome', 'age', 'gender', 'Admission time', 'Discharge time']
    biomarkers = [col for col in group.columns if col not in exclude]

    result = {}
    for col in biomarkers:
        valid_mask = group[col].notna()
        if valid_mask.any():
            w = weights[valid_mask]
            val = group.loc[valid_mask, col]
            result[col] = np.sum(val * w) / np.sum(w)
        else:
            result[col] = np.nan
            
    result['outcome'] = group['outcome'].iloc[0]
    result['RE_DATE'] = last_date.date()
    if 'age' in group.columns and 'gender' in group.columns:
        result['age'] = group['age'].iloc[0]
        result['gender'] = group['gender'].iloc[0]
    
    return pd.Series(result)

def preprocess_data(test_csv_file, output_csv_file):
    df = pd.read_csv(test_csv_file)
    
    df["PATIENT_ID"] = df["PATIENT_ID"].ffill()
    df["RE_DATE"] = pd.to_datetime(df["RE_DATE"], errors='coerce').dt.date

    df = df.groupby('PATIENT_ID').apply(time_weighted_average, include_groups=False).reset_index()
    df = df.drop(columns=["PATIENT_ID", "RE_DATE"])

    df.to_csv(output_csv_file, index=False)

def data_enrichment():
    data1 = pd.read_csv('110_preprocessed.csv')
    data2 = pd.read_csv('375_preprocessed.csv')

    common_cols = ['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte', 'outcome']

    data1 = data1[common_cols]
    data2 = data2[common_cols]

    df_enriched = pd.concat([data1, data2], ignore_index=True)
    df_enriched.to_csv('enriched_data.csv', index=False)

def handle_outliers(df, cols):
    for col in cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    return df

def clean_and_impute(df):
    df = df.dropna(subset=['outcome'])
    if 'age' in df.columns and 'gender' in df.columns:
        df = df.dropna(subset=['age', 'gender'])
    limit_col = len(df) * 0.5
    df = df.dropna(thresh=limit_col, axis=1)
    exclude = ['outcome', 'age', 'gender']
    cols_to_fix = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=cols_to_fix, how='all')
    imputer = SimpleImputer(strategy='median')
    df[cols_to_fix] = imputer.fit_transform(df[cols_to_fix])
    df = handle_outliers(df, cols_to_fix)
    df = df.round(3)
    return df

preprocess_data('test_data_110.csv', '110_preprocessed.csv')
preprocess_data('train_data_375.csv', '375_preprocessed.csv')
data_enrichment()
clean_and_impute(pd.read_csv('110_preprocessed.csv')).to_csv('./final_data/110_cleaned.csv', index=False)
clean_and_impute(pd.read_csv('enriched_data.csv')).to_csv('./final_data/enriched_data_cleaned.csv', index=False)
clean_and_impute(pd.read_csv('375_preprocessed.csv')).to_csv('./final_data/375_cleaned.csv', index=False)
