

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("https://dagshub.com/lsLohith1414/Fraud_Detection.mlflow")

mlflow.set_experiment("Network_security")


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


with mlflow.start_run(run_name="logistic",description="just exp"):

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_metric("accurach",acc)
    mlflow.log_param("max_iter", 200)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="iris_classifier_logistic"
    )
print("Run logged to DagsHub successfully!")