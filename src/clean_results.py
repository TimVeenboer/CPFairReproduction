import pandas as pd
### CHANGE - ENTIRE FILE
    
def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    # calculates and creates all extra needed columns
    w = 0.5
    df['DCF'] = abs((df['ndcg_ACT']-df['ndcg_INACT'])/(df['ndcg_ACT']+df['ndcg_INACT']))
    df['Cov.'] = df['Cov_ALL']/100
    df['All_Items'] = df['All_Items'].str.extract(r'(\d+)==\d+\.0').astype(int)
    df['Short.'] = df['Short_Items'] / df['All_Items']
    df['Long.'] = df['Long_Items'] / df['All_Items']
    df["DPF"] = df['Short.'] - df['Long.']
    df['mCPF'] = w * df['DPF'] + (1 - w) * df['DCF']
    df['mCPF/All'] = df['mCPF'] / df['ndcg_ALL']
    df['delta (%)'] = ((df['mCPF'].iloc[0] - df['mCPF']) / df['mCPF'].iloc[0]) * 100

    # cleans the dataframe by renaming and reodering the columns as in the paper
    df = df.rename(columns={'ndcg_ALL': 'All',
                    'ndcg_ACT': 'Active',
                    'ndcg_INACT': 'Inactive',
                    'Nov_ALL': 'Nov.',
                    })
    df = df[['Dataset', 'Model', 'Type', 'All', 'Active', 'Inactive', 'DCF', 'Nov.', 'Cov.', 'Short.', 'Long.', 'DPF', 'mCPF', 'mCPF/All', 'delta (%)']]

    # Rounding all columns except the last one to four decimal places
    df.iloc[:, :-1] = df.iloc[:, :-1].round(4)
    # Rounding the last column to two decimal places
    df.iloc[:, -1] = df.iloc[:, -1].round(2)
    
    return df
