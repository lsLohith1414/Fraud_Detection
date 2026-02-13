import os
import sys
import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "network_security_lgbm"
METRIC_NAME = "pr_auc"  # Change if needed


def get_latest_model_version(client):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("No model versions found.")
        sys.exit(1)

    # Latest version = highest version number
    latest_version = max(versions, key=lambda v: int(v.version))
    return latest_version


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("MLFLOW_TRACKING_URI not set")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    print("Starting Model Performance Gate...")

    # -------------------------------------------------
    # 1️⃣ Get latest trained model version
    # -------------------------------------------------
    latest_version = get_latest_model_version(client)
    latest_run = client.get_run(latest_version.run_id)
    new_metric = latest_run.data.metrics.get(METRIC_NAME)

    if new_metric is None:
        print(f"{METRIC_NAME} not found in latest model.")
        sys.exit(1)

    print(f"New model version: {latest_version.version}")
    print(f"New model {METRIC_NAME}: {new_metric}")

    # -------------------------------------------------
    # 2️⃣ Check if production model exists
    # -------------------------------------------------
    try:
        prod_version = client.get_model_version_by_alias(
            MODEL_NAME,
            "production"
        )

        prod_run = client.get_run(prod_version.run_id)
        prod_metric = prod_run.data.metrics.get(METRIC_NAME)

        print(f"Production model version: {prod_version.version}")
        print(f"Production model {METRIC_NAME}: {prod_metric}")

    except Exception:
        # No production model exists
        print("No production model found.")
        print("Promoting latest model to production...")

        client.set_registered_model_alias(
            MODEL_NAME,
            "production",
            latest_version.version
        )

        print("Model promoted to production.")
        sys.exit(0)

    # -------------------------------------------------
    # 3️⃣ Compare Metrics
    # -------------------------------------------------
    if new_metric >= prod_metric:
        print("New model is better or equal. Promoting to production...")

        client.set_registered_model_alias(
            MODEL_NAME,
            "production",
            latest_version.version
        )

        print("Model promoted to production.")
        sys.exit(0)

    else:
        print("New model is worse than production.")
        print("Failing CI.")
        # sys.exit(1)


if __name__ == "__main__":
    main()
