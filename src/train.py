"""
src/train.py

CloudCost Sentinel V1 training script.

Simple flow:
load data
    ↓
create features
    ↓
train / validation / test split
    ↓
choose model
    ↓
train
    ↓
evaluate
    ↓
MLflow

Optuna tuning is implemented for XGBoost.
"""

import os

# hide TensorFlow startup noise; has to be set before keras imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import xgboost as xgb
from mlflow import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from src.features import add_features, chronological_split, prepare_xy
from src.evaluate import evaluate_model, print_metrics, find_best_threshold


DATA_PATH = "data/cloud_cost.csv"
MODEL_DIR = "models/cloudcost"
MLFLOW_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "CloudCostSentinel"


# load + prepare data

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]) # converts back to pd datetime format so that .dt.dayofweek can be used

    print("Dataset shape:", df.shape)
    print(df.head())

    print("\nAnomaly distribution:")
    print(df["is_anomaly"].value_counts())

    df = add_features(df)

    train_df, val_df, test_df = chronological_split(df)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_xy(
        train_df,
        val_df,
        test_df,
    )

    print(f"\nChronological Split Sizes -> Train: {len(train_df):,}, Validation: {len(val_df):,}, Test: {len(test_df):,}")
    print(f"Positive examples -> Train: {int(y_train.sum()):,}, Validation: {int(y_val.sum()):,}, Test: {int(y_test.sum()):,}")

    return (
        train_df, val_df, test_df,
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )


# common MLflow logging

def log_metrics_to_mlflow(val_metrics, test_metrics):
    """
    Same metric logging for Logistic Regression, XGBoost and NN.
    """

    mlflow.log_metrics({
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_anomaly_rate": val_metrics["anomaly_rate"],
        "val_predicted_anomaly_rate": val_metrics["predicted_anomaly_rate"],

        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_anomaly_rate": test_metrics["anomaly_rate"],
        "test_predicted_anomaly_rate": test_metrics["predicted_anomaly_rate"],
    })


# logistic regression

def train_logistic_regression(
    X_train, X_val, X_test,
    y_train, y_val, y_test
):
    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced", #It changes how much each class matters during optimization.
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]

    best_threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    val_metrics = evaluate_model(
        y_val,
        val_probabilities,
        threshold=best_threshold,
    )

    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_model(
        y_test,
        test_probabilities,
        threshold=best_threshold,
    )

    print_metrics(val_metrics, "Logistic Regression Validation")
    print_metrics(test_metrics, "Logistic Regression Test")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl",
    )

    joblib.dump(model, model_path)

    # mlflow
    mlflow.set_experiment("CloudCost_LogisticRegression")

    try:
        with mlflow.start_run(run_name="logistic_regression_baseline"):

            mlflow.log_params({
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "threshold": best_threshold,
            })

            log_metrics_to_mlflow(
                val_metrics,
                test_metrics,
            )

            mlflow.log_artifact(model_path)

    except Exception as e:
        print("\nMLflow logging skipped.")
        print("Reason:", e)


# baseline XGBoost

def train_xgboost(
    X_train, X_val, X_test,
    y_train, y_val, y_test
):
    print("\nTraining XGBoost...")

    # anomaly class is much smaller, so give it more weight using train data only
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    xgb_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "eval_metric": "logloss",
    }

    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(**xgb_params)

    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]

    best_threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    val_metrics = evaluate_model(
        y_val,
        val_probabilities,
        threshold=best_threshold,
    )

    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_model(
        y_test,
        test_probabilities,
        threshold=best_threshold,
    )

    print_metrics(val_metrics, "XGBoost Validation")
    print_metrics(test_metrics, "XGBoost Test")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "xgb_cloudcost_model.pkl",
    )

    joblib.dump(model, model_path)

    joblib.dump(
        X_train.columns.tolist(),
        os.path.join(MODEL_DIR, "feature_columns.pkl"),
    )

    with open(
        os.path.join(MODEL_DIR, "threshold.txt"),
        "w",
    ) as file:
        file.write(str(best_threshold))

    # mlflow
    mlflow.set_experiment("CloudCost_XGBoost")

    try:
        with mlflow.start_run(run_name="xgb_baseline"):

            mlflow.log_params({
                **xgb_params,
                "threshold": best_threshold,
            })

            log_metrics_to_mlflow(
                val_metrics,
                test_metrics
            )

            mlflow.log_artifact(model_path)

    except Exception as e:
        print("\nMLflow logging skipped.")
        print("Reason:", e)


# neural network

def train_neural_network(
    X_train, X_val, X_test,
    y_train, y_val, y_test
):
    print("\nTraining Neural Network...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(
            64,
            activation="relu",
            input_shape=(X_train_scaled.shape[1],),
        ),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    val_probabilities = model.predict(
        X_val_scaled,
        verbose=0,
    ).flatten()

    best_threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    val_metrics = evaluate_model(
        y_val,
        val_probabilities,
        threshold=best_threshold,
    )

    test_probabilities = model.predict(
        X_test_scaled,
        verbose=0,
    ).flatten()

    test_metrics = evaluate_model(
        y_test,
        test_probabilities,
        threshold=best_threshold,
    )

    print_metrics(val_metrics, "Neural Network Validation")
    print_metrics(test_metrics, "Neural Network Test")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "nn_cloudcost_model.keras",
    )

    scaler_path = os.path.join(
        MODEL_DIR,
        "nn_scaler.pkl",
    )

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # mLflow
    mlflow.set_experiment("CloudCost_NN")

    try:
        with mlflow.start_run(run_name="nn_baseline"):

            mlflow.log_params({
                "hidden_layers": "64-32-16",
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "epochs": 50,
                "batch_size": 32,
                "threshold": best_threshold,
            })

            log_metrics_to_mlflow(
                val_metrics,
                test_metrics
            )

            mlflow.log_artifact(model_path)
            mlflow.log_artifact(scaler_path)

    except Exception as e:
        print("\nMLflow logging skipped.")
        print("Reason:", e)

# Optuna tuned XGBoost
# def objective(trial):
#     1. trial suggests XGBoost hyperparameters
#     2. create XGBClassifier(**params)
#     3. fit on X_train
#     4. predict probabilities on X_val
#     5. calculate a validation score
#     6. return that score
#
# study = optuna.create_study(...)
# study.optimize(objective, n_trials=...)
def tune_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    # same imbalance weight for every trial; calculate from train only
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    print(f"Optuna XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    # optuna objective: one trial = one hyperparameter configuration + validation score
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "logloss"
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        val_probabilities = model.predict_proba(X_val)[:, 1]  # probability of the validation row is class 1

        best_threshold = find_best_threshold(y_val, val_probabilities)

        metrics = evaluate_model(
            y_val,
            val_probabilities,
            threshold=best_threshold
        )

        return metrics["f1"]

    # run optuna
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=25)

    print("Best parameters:")
    print(study.best_params)

    print("Best F1:")
    print(study.best_value)

    # final model
    best_params = study.best_params.copy()

    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = 42
    best_params["eval_metric"] = "logloss"

    model = xgb.XGBClassifier(**best_params)

    model.fit(X_train, y_train)

    # validation
    val_probabilities = model.predict_proba(X_val)[:, 1]

    best_threshold = find_best_threshold(y_val, val_probabilities)

    val_metrics = evaluate_model(y_val, val_probabilities, threshold=best_threshold)

    print_metrics(val_metrics, "Optuna XGBoost Validation")

    # test
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_model(y_test, test_probabilities, threshold=best_threshold)

    print_metrics(test_metrics, "Optuna XGBoost Test")

    # save tuned model + serving info
    os.makedirs(MODEL_DIR, exist_ok=True)  # creates folder if it doesnt exist

    model_path = os.path.join(MODEL_DIR, "xgb_optuna_model.pkl")

    joblib.dump(model, model_path)  # save the already-trained model instance

    # FastAPI also needs the exact train columns and validation-selected threshold
    feature_columns_path = os.path.join(
        MODEL_DIR,
        "xgb_optuna_feature_columns.pkl",
    )
    threshold_path = os.path.join(
        MODEL_DIR,
        "xgb_optuna_threshold.txt",
    )

    joblib.dump(X_train.columns.tolist(), feature_columns_path)

    with open(threshold_path, "w") as file:
        file.write(str(best_threshold))

    # MLflow + model registry
    mlflow.set_experiment("Cloudcost_tunedXGB")
    try:
        with mlflow.start_run(run_name="tuned_xgb_final"):
            mlflow.log_params({
                **study.best_params,
                "scale_pos_weight": scale_pos_weight,
                "threshold": best_threshold,
                "optuna_trials": 25,
            })
            log_metrics_to_mlflow(val_metrics, test_metrics)

            # keep these because FastAPI still uses them for feature order + threshold
            mlflow.log_artifact(feature_columns_path)
            mlflow.log_artifact(threshold_path)

            # log the trained XGB in MLflow's model format, then register this run's model
            model_info = mlflow.xgboost.log_model(xgb_model=model, name="model")
            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=REGISTERED_MODEL_NAME,
            )

            # champion = the version we currently want to serve
            client = MlflowClient()
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="champion",
                version=model_version.version,
            )

            print(f"Registered {REGISTERED_MODEL_NAME} v{model_version.version} as champion")

    except Exception as e:
        print("\nMLflow logging/registry skipped.")
        print("Reason:", e)
    
if __name__ == "__main__":
    main()


# choose what to train

def main():

    mlflow.set_tracking_uri(MLFLOW_URI)

    (
        train_df, val_df, test_df,
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = load_data()

    print("\nChoose a model:")
    print("1 - Logistic Regression")
    print("2 - XGBoost")
    print("3 - Neural Network")
    print("4 - Run all")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        train_logistic_regression(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )

    elif choice == "2":

        tune_choice = input(
            "\nUse Optuna tuning? (y/n): "
        ).strip().lower()

        if tune_choice == "y":
            tune_xgboost(
                X_train, X_val, X_test,
                y_train, y_val, y_test
            )

        else:
            train_xgboost(
                X_train, X_val, X_test,
                y_train, y_val, y_test
            )

    elif choice == "3":
        train_neural_network(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )

    elif choice == "4":
        train_logistic_regression(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )

        train_xgboost(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )

        train_neural_network(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )

    else:
        print("Invalid choice.")
...
