import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        # If you need constants, define here
        self.high_risk_categories = {"electronics", "jewelry", "crypto"}

    def fit(self, X, y=None):
        # No fitting required for feature engineering
        return self

    def transform(self, X):
        df = X.copy()

        # --------------------------------------------------
        # 1️⃣ TIME PARSING & SORTING
        # --------------------------------------------------
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")

        df = df.dropna(subset=["Timestamp"])
        df = df.sort_values(["CustomerID", "Timestamp"]).reset_index(drop=True)

        # --------------------------------------------------
        # 2️⃣ BASIC TIME FEATURES
        # --------------------------------------------------
        df["txn_hour"] = df["Timestamp"].dt.hour
        df["txn_weekday"] = df["Timestamp"].dt.weekday
        df["txn_month"] = df["Timestamp"].dt.month
        df["is_weekend"] = df["txn_weekday"].isin([5, 6]).astype(int)

        df["days_since_last_login"] = (
            df["Timestamp"] - df["LastLogin"]
        ).dt.days

        df["txn_gap_minutes"] = (
            df.groupby("CustomerID")["Timestamp"]
            .diff()
            .dt.total_seconds()
            .div(60)
        )

        # --------------------------------------------------
        # 3️⃣ USER BEHAVIOR FEATURES
        # --------------------------------------------------
        df["user_avg_amount"] = (
            df.groupby("CustomerID")["TransactionAmount"]
            .expanding()
            .mean()
            .shift()
            .reset_index(level=0, drop=True)
        )

        df["amount_diff_from_avg"] = (
            df["TransactionAmount"] - df["user_avg_amount"]
        )

        df["amount_ratio_avg"] = (
            df["TransactionAmount"] / (df["user_avg_amount"] + 1)
        )

        df["user_txn_count"] = df.groupby("CustomerID").cumcount()

        df["date"] = df["Timestamp"].dt.date
        df["user_daily_txns"] = (
            df.groupby(["CustomerID", "date"]).cumcount()
        )

        # --------------------------------------------------
        # 4️⃣ VELOCITY FEATURES
        # --------------------------------------------------
        df["txns_last_1h"] = 0
        df["txns_last_24h"] = 0
        df["amount_last_24h"] = 0.0

        for cid, group in df.groupby("CustomerID"):
            times = group["Timestamp"].values
            amounts = group["TransactionAmount"].values

            txns_1h = np.zeros(len(group), dtype=int)
            txns_24h = np.zeros(len(group), dtype=int)
            amt_24h = np.zeros(len(group), dtype=float)

            for i in range(len(group)):
                t = times[i]
                mask_1h = (times < t) & (
                    times >= t - np.timedelta64(1, "h")
                )
                mask_24h = (times < t) & (
                    times >= t - np.timedelta64(24, "h")
                )

                txns_1h[i] = mask_1h.sum()
                txns_24h[i] = mask_24h.sum()
                amt_24h[i] = amounts[mask_24h].sum()

            df.loc[group.index, "txns_last_1h"] = txns_1h
            df.loc[group.index, "txns_last_24h"] = txns_24h
            df.loc[group.index, "amount_last_24h"] = amt_24h

        # --------------------------------------------------
        # 5️⃣ MERCHANT FEATURES
        # --------------------------------------------------
        df = df.sort_values(["MerchantID", "Timestamp"])

        df["merchant_txn_count"] = df.groupby("MerchantID").cumcount()

        df["merchant_avg_amount"] = (
            df.groupby("MerchantID")["TransactionAmount"]
            .expanding()
            .mean()
            .shift()
            .reset_index(level=0, drop=True)
        )

        df["amount_diff_from_merchant_avg"] = (
            df["TransactionAmount"] - df["merchant_avg_amount"]
        )

        # --------------------------------------------------
        # 6️⃣ LOCATION FEATURES
        # --------------------------------------------------
        df["is_new_location"] = (
            df.groupby("CustomerID")["Location"]
            .transform(lambda x: (~x.duplicated()).astype(int))
        )

        # --------------------------------------------------
        # 7️⃣ CATEGORY RISK
        # --------------------------------------------------
        df["is_high_risk_category"] = (
            df["Category"].isin(self.high_risk_categories).astype(int)
        )

        return df
