# import os 
# import pandas as pd

# from src.common.logger import get_logger
# logger = get_logger(__name__)
# from src.common.exception import CustomException


# from src.entities.config.data_transformation_config import DataTransformationConfig

# path_train_df_path = os.path.join("artifacts","01_28_2026_14_15","data_validation","valid","train.csv")

# train_df = pd.read_csv(path_train_df_path)

# print(train_df)


# def create_new_features_from_existing(df:pd.DataFrame) -> pd.DataFrame:
#     pass


     
# def create_new_features_from_existing_features(df:pd.DataFrame) -> pd.DataFrame:
#     # 1. Time based featues:

#     df["txn_hour"]     = df["Timestamp"].dt.hour
#     df["txn_weekday"]  = df["Timestamp"].dt.weekday   # 0=Mon
#     df["txn_month"]    = df["Timestamp"].dt.month
#     df["is_weekend"]   = df["txn_weekday"].isin([5,6]).astype(int)


#     ## 2. USER-LEVEL BEHAVIOR FEATURES

#     df['user_avg_amount'] = df.groupby('CustomerID')['TransactionAmount'].transform('mean')
#     df['amount_diff_from_avg'] = df['TransactionAmount'] - df['user_avg_amount']
#     df['amount_ratio_avg'] = df['TransactionAmount'] / (df['user_avg_amount'] + 1)
#     df['user_txn_count'] = df.groupby('CustomerID').cumcount() + 1
#     df['date'] = pd.to_datetime(df['Timestamp']).dt.date
#     df['user_daily_txns'] = df.groupby(['CustomerID', 'date']).cumcount() + 1



#         ## 3. VELOCITY FEATURES
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#     df = df.sort_values(['CustomerID', 'Timestamp'])




#     # Transactions in the Past 1 Hour
#     # Convert timestamp
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#     df = df.dropna(subset=['Timestamp'])
    
#     # Sort properly
#     df = df.sort_values(['CustomerID', 'Timestamp']).reset_index(drop=True)
    
#     # Empty column
#     df['txns_last_1h'] = 0
    
#     # Loop per customer (FAST because each customer has few transactions)
#     for cid, group in df.groupby('CustomerID'):
#         times = group['Timestamp'].values
#         result = np.zeros(len(times), dtype=int)
        
#         for i in range(len(times)):
#             # find how many timestamps fall within [current_time - 1h, current_time]
#             cutoff = times[i] - np.timedelta64(1, 'h')
#             result[i] = np.sum((times >= cutoff) & (times <= times[i]))
        
#         df.loc[group.index, 'txns_last_1h'] = result



#     # Transactions in the Past 24 Hours
#     # Make sure timestamps are clean
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#     df = df.dropna(subset=['Timestamp'])
    
#     # Sort properly
#     df = df.sort_values(['CustomerID', 'Timestamp']).reset_index(drop=True)

#     # Create empty columns
#     df['txns_last_24h'] = 0
#     df['amount_last_24h'] = 0.0
    
#     # Loop per customer (FAST because each customer has few transactions)
#     for cid, group in df.groupby('CustomerID'):
#         times = group['Timestamp'].values
#         amounts = group['TransactionAmount'].values
    
#         txn_count = np.zeros(len(times), dtype=int)
#         amount_sum = np.zeros(len(times), dtype=float)
    
#         for i in range(len(times)):
#             cutoff = times[i] - np.timedelta64(24, 'h')
    
#             mask = (times >= cutoff) & (times <= times[i])
    
#             txn_count[i] = np.sum(mask)
#             amount_sum[i] = np.sum(amounts[mask])
    
#         df.loc[group.index, 'txns_last_24h'] = txn_count
#         df.loc[group.index, 'amount_last_24h'] = amount_sum



#     # Amount spent in the last 24 hours
#     # Ensure timestamp is cleaned and sorted
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#     df = df.dropna(subset=['Timestamp'])
    
#     df = df.sort_values(['CustomerID', 'Timestamp']).reset_index(drop=True)
    
#     # Create empty output column
#     df['amount_last_24h'] = 0.0
    
#     # Loop per customer ID
#     for cid, group in df.groupby('CustomerID'):
#         times = group['Timestamp'].values
#         amounts = group['TransactionAmount'].values
    
#         amount_sum_24h = np.zeros(len(times), dtype=float)
    
#         for i in range(len(times)):
#             cutoff = times[i] - np.timedelta64(24, 'h')
    
#             # Mask of transactions within [current_time - 24h, current_time]
#             mask = (times >= cutoff) & (times <= times[i])
    
#             amount_sum_24h[i] = np.sum(amounts[mask])
    
#         # Assign back to main df
#         df.loc[group.index, 'amount_last_24h'] = amount_sum_24h



#     ## 4. MERCHANT-LEVEL RISK FEATURES
    
#     df['merchant_txn_count'] = df.groupby('MerchantID')['TransactionAmount'].transform('count')
#     df['merchant_avg_amount'] = df.groupby('MerchantID')['TransactionAmount'].transform('mean')
#     df['amount_diff_from_merchant_avg'] = df['TransactionAmount'] - df['merchant_avg_amount']
    
    
    
    
    
#     # 5. LOCATION & DEVICE FEATURES
    
#     df['is_new_location'] = (
#         df.groupby('CustomerID')['Location']
#               .transform(lambda x: ~x.duplicated().astype(int))
#     )
    
    
#     # 6. CATEGORY RISK FEATURES
    
#     high_risk = ['electronics','jewelry','crypto']
#     df['is_high_risk_category'] = df['Category'].isin(high_risk).astype(int)



