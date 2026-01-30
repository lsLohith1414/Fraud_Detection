import os 
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from imblearn.combine import SMOTEENN


from src.common.logger import get_logger
logger = get_logger(__name__)
from src.common.exception import CustomException
from src.common.utils import save_preprocessor


from src.entities.config.data_transformation_config import DataTransformationConfig
from src.entities.artifact.artifacts_entity import  DataValidationArtifact, DataTransformationArtifact




class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig, data_validation_artifact:DataValidationArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact

    # step-1: Creating new features
    # New feature creation function using existing features 
    def create_new_features_from_existing_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Leakage-safe feature engineering for transactional / fraud data.
        All cumulative features use ONLY past information.
        """

        df = df.copy()

        # ------------------------------------------------------------------
        # 1️⃣ TIME PARSING & SORTING  (CRITICAL FOR LEAKAGE SAFETY)
        # ------------------------------------------------------------------
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")

        df = df.dropna(subset=["Timestamp"])
        df = df.sort_values(["CustomerID", "Timestamp"]).reset_index(drop=True)

        # ------------------------------------------------------------------
        # 2️⃣ BASIC TIME FEATURES (NO LEAKAGE)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 3️⃣ USER-LEVEL BEHAVIOR FEATURES (LEAKAGE-SAFE)
        # ------------------------------------------------------------------
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

        df["user_txn_count"] = (
            df.groupby("CustomerID").cumcount()
        )

        df["date"] = df["Timestamp"].dt.date

        df["user_daily_txns"] = (
            df.groupby(["CustomerID", "date"])
            .cumcount()
        )

        # ------------------------------------------------------------------
        # 4️⃣ VELOCITY FEATURES (PAST-ONLY WINDOWS)
        # ------------------------------------------------------------------
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
                mask_1h = (times < t) & (times >= t - np.timedelta64(1, "h"))
                mask_24h = (times < t) & (times >= t - np.timedelta64(24, "h"))

                txns_1h[i] = mask_1h.sum()
                txns_24h[i] = mask_24h.sum()
                amt_24h[i] = amounts[mask_24h].sum()

            df.loc[group.index, "txns_last_1h"] = txns_1h
            df.loc[group.index, "txns_last_24h"] = txns_24h
            df.loc[group.index, "amount_last_24h"] = amt_24h

        # ------------------------------------------------------------------
        # 5️⃣ MERCHANT-LEVEL FEATURES (LEAKAGE-SAFE)
        # ------------------------------------------------------------------
        df = df.sort_values(["MerchantID", "Timestamp"])

        df["merchant_txn_count"] = (
            df.groupby("MerchantID").cumcount()
        )

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

        # ------------------------------------------------------------------
        # 6️⃣ LOCATION & DEVICE FEATURES
        # ------------------------------------------------------------------
        df["is_new_location"] = (
            df.groupby("CustomerID")["Location"]
            .transform(lambda x: (~x.duplicated()).astype(int))
        )

        # ------------------------------------------------------------------
        # 7️⃣ CATEGORY RISK FEATURES
        # ------------------------------------------------------------------
        high_risk_categories = {"electronics", "jewelry", "crypto"}
        df["is_high_risk_category"] = (
            df["Category"].isin(high_risk_categories).astype(int)
        )

        return df
    


    # # Handle null values
    def handle_null_values(self,df:pd.DataFrame)->pd.DataFrame:
        
        df['txn_gap_minutes'] = df['txn_gap_minutes'].fillna(1440)
        return df
    



    # Remove the useless features and keep the features that are used for modeling
    def remove_ueless_features(self,df:pd.DataFrame)->pd.DataFrame:
        
        drop_cols = [
        'TransactionID', 'CustomerID', 'MerchantID', 'MerchantName',
        'Name', 'Address', 'Location', 'Timestamp', 'LastLogin', 'date', 'SuspiciousFlag'
        ]
        
        df = df.drop(columns=drop_cols)
        return df
    

    # step-2 proprocessing Encoding, Scaling, oversampling data
    # 1. Cyclic encoding for those are created using date time feaures

    def cyclic_encoding(self,df:pd.DataFrame)->pd.DataFrame:

        # CYCLIC ENCODING (on train, then test) on time based columns


        df['txn_hour_sin'] = np.sin(2*np.pi*df['txn_hour']/24)
        df['txn_hour_cos'] = np.cos(2*np.pi*df['txn_hour']/24)

        df['txn_weekday_sin'] = np.sin(2*np.pi*df['txn_weekday']/7)
        df['txn_weekday_cos'] = np.cos(2*np.pi*df['txn_weekday']/7)

        df['txn_month_sin'] = np.sin(2*np.pi*df['txn_month']/12)
        df['txn_month_cos'] = np.cos(2*np.pi*df['txn_month']/12)

        # drop raw time columns
        time_cols = ['txn_hour', 'txn_weekday', 'txn_month']
        df = df.drop(columns=time_cols)
        
        
        return df
    


    # 2. one hot necoder or categorical columns
    def one_hot_encode_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
        """
        Fit OneHotEncoder on training data and apply the same transformation on test data.
        """

        ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        # Fit ONLY on train
        train_array = ohe.fit_transform(train_df[categorical_cols])

        # Transform ONLY on test
        test_array = ohe.transform(test_df[categorical_cols])

        ohe_cols = ohe.get_feature_names_out(categorical_cols)

        train_ohe_df = pd.DataFrame(
            train_array,
            columns=ohe_cols,
            index=train_df.index
        )

        test_ohe_df = pd.DataFrame(
            test_array,
            columns=ohe_cols,
            index=test_df.index
        )

        train_df_encoded = pd.concat(
            [train_df.drop(columns=categorical_cols), train_ohe_df],
            axis=1
        )

        test_df_encoded = pd.concat(
            [test_df.drop(columns=categorical_cols), test_ohe_df],
            axis=1
        )

        return train_df_encoded, test_df_encoded, ohe


    def null_values_handles(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values:
        - Numerical columns -> mean
        - Categorical columns -> mode
        """

        df_cleaned = df.copy()

        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:

                # Numerical columns
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(
                        df_cleaned[col].mean()
                    )

                # Categorical columns
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(
                        df_cleaned[col].mode()[0]
                    )

        return df_cleaned


    def scale_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        robust_features: List[str],
        minmax_features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
        """
        Fit scalers ONLY on training data and apply them to both
        training and test datasets.

        Returns:
            - Scaled train_df
            - Scaled test_df
            - Dict of fitted scalers (for persistence)
        """

        train_df_scaled = train_df.copy()
        test_df_scaled = test_df.copy()

        scalers = {}

        # -------------------- RobustScaler --------------------
        if robust_features:
            robust_scaler = RobustScaler()
            train_df_scaled[robust_features] = robust_scaler.fit_transform(
                train_df[robust_features]
            )
            test_df_scaled[robust_features] = robust_scaler.transform(
                test_df[robust_features]
            )
            scalers["robust_scaler"] = robust_scaler
            scalers["robust_features"] = robust_features

        # -------------------- MinMaxScaler --------------------
        if minmax_features:
            minmax_scaler = MinMaxScaler()
            train_df_scaled[minmax_features] = minmax_scaler.fit_transform(
                train_df[minmax_features]
            )
            test_df_scaled[minmax_features] = minmax_scaler.transform(
                test_df[minmax_features]
            )
            scalers["minmax_scaler"] = minmax_scaler
            scalers["minmax_features"] = minmax_features

        return train_df_scaled, test_df_scaled, scalers
    
    # 3. over Sampling function uses SMOTE-ENN

    def oversample_smoteenn(
        self,
        train_df: pd.DataFrame,
        target_column: str
    ) -> pd.DataFrame:
        """
        Apply SMOTE-ENN oversampling ONLY on training data
        and return a single resampled training DataFrame.
        """

        # Split features and target
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        # Apply SMOTE-ENN
        smote_enn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

        # Combine back into a single DataFrame
        train_resampled = pd.concat(
            [
                pd.DataFrame(X_resampled, columns=X_train.columns),
                pd.Series(y_resampled, name=target_column)
            ],
            axis=1
        )

        return train_resampled
    

    def build_preprocessor(
        self,
        ohe,
        scaler: dict,
        categorical_cols: list
    ) -> dict:
        """
        Build a single preprocessor object from fitted encoder and scalers.
        """

        preprocessor = {
            # One-hot encoding
            "one_hot_encoder": ohe,
            "categorical_cols": categorical_cols,

            # Scalers (already fitted)
            "robust_scaler": scaler.get("robust_scaler"),
            "robust_features": scaler.get("robust_features", []),

            "minmax_scaler": scaler.get("minmax_scaler"),
            "minmax_features": scaler.get("minmax_features", [])
        }

        return preprocessor







        


    
    def initiate_data_transformation(self):

        try: 

            train_df_path = self.data_validation_artifact.valid_train_file_path
            test_df_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(train_df_path)
            test_df = pd.read_csv(test_df_path)

            # step-1: adding features
            # creating new feature

            logger.info("Data Transformation Started")

            logger.info("Adding Engineered features step started ")
            train_df = self.create_new_features_from_existing_features(train_df)
            test_df = self.create_new_features_from_existing_features(test_df)
            logger.info(f"Adding Engineered features step successfully completed [{train_df.columns.tolist()}]")



            # handling null values
            logger.info("Handling null value step started")
            train_df = self.handle_null_values(train_df)
            test_df = self.handle_null_values(test_df)
            logger.info("Handling null value step successfully completed")



            # Remove the useless features
            logger.info("removing useless feature step started")
            train_df = self.remove_ueless_features(train_df)
            test_df = self.remove_ueless_features(test_df)
            logger.info(f"removing useless feature step successfully completed [{train_df.columns.tolist()}]")



            # step-2 preprocessing
            # 1. Cyclic encoding
            logger.info("Cyclic encoding step started")
            train_df = self.cyclic_encoding(train_df)
            test_df = self.cyclic_encoding(test_df)
            logger.info(f"Cyclic encoding step successfully completed [{train_df.columns.tolist()}]")



            # 2. ohe
            logger.info("One hot encoding step started")
            categorical_cols = ["Category"]
            train_df_encoded, test_df_encoded, ohe = self.one_hot_encode_train_test(train_df,test_df,categorical_cols)
            logger.info(f"One hot encoding step successfully completed [{train_df_encoded.columns.tolist()}]")


            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path))

            




            # 3. Scaling 
            # usage
            ROBUST_FEATURES = [
                "Amount", "TransactionAmount", "AccountBalance",
                "user_avg_amount", "amount_last_24h",
                "merchant_avg_amount",
                "amount_diff_from_avg",
                "amount_diff_from_merchant_avg",
                "user_txn_count", "user_daily_txns",
                "txns_last_1h", "txns_last_24h",
                "merchant_txn_count",
                "days_since_last_login", "txn_gap_minutes"
            ]

            MINMAX_FEATURES = [
                "AnomalyScore", "amount_ratio_avg", "Age"
            ]

            logger.info("Feature scaling step started")
            train_df_scaled, test_df_scaled, scalers = self.scale_train_test(
                train_df=train_df_encoded,
                test_df=test_df_encoded,
                robust_features=ROBUST_FEATURES,
                minmax_features=MINMAX_FEATURES
            )




            preprocessor = self.build_preprocessor(ohe,scalers,["Category"])
            save_preprocessor(preprocessor,self.data_transformation_config.preprocessor_file_path)








            logger.info(f"Feature scaling step successfully completed")

            # 3.1 null value handle once again 
            test_df_scaled = self.null_values_handles(test_df_scaled)
            train_df_scaled = self.null_values_handles(train_df_scaled)

            test_df_scaled.to_csv(self.data_transformation_config.transformed_test_file_path,index = False)



            


            

            # os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path))

            # test_df_scaled.to_csv(self.data_transformation_config.transformed_test_file_path)
            # logger.info(f"Feature scaler test path : [{self.data_transformation_config.transformed_test_file_path}]")


            # os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_file_path))

            # save_preprocessor(file_path=self.data_transformation_config.preprocessor_file_path)


            # 4. oversampling
            logger.info("over sampling step started ")
            train_df_resampled = self.oversample_smoteenn(train_df_scaled,"FraudIndicator")

            train_df_resampled.to_csv(self.data_transformation_config.transformed_train_file_path,index=False)


            # abc = syke()
            logger.info("Transformation step completed successfully")


            self.data_transformation_artifact = DataTransformationArtifact(transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                                                                      transformed_test_file_path= self.data_transformation_config.transformed_test_file_path,
                                                                      transformed_object_file_path=self.data_transformation_config.preprocessor_file_path)


            return self.data_transformation_artifact
        
        except Exception as e:
            logger.error("Data Transformation pipeline failed", exc_info=True)
            raise CustomException(e)

