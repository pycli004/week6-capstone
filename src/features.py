import pandas as pd

def engineer_features(df):

    df['TenureBucket'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, 100],
        labels=['0-1yr', '1-2yr', '2-4yr', '4yr+']
    ).astype(str)

        
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    def count_services(row):
        count = 0
        for col in service_cols:
            val = str(row[col])
            if val not in ['No', 'No internet service', 'No phone service', 'nan']:
                count += 1
        return count
    
    df['TotalServices'] = df[service_cols].apply(count_services, axis=1)

    df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    df['ContractRisk'] = (df['Contract'] == 'Month-to-month').astype(int)

    df['HasFamily'] = (
        (df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')
    ).astype(int)
    
    return df