"""
Week 9: predicting new people with the refit pipeline.

The new rows go through exactly the same steps as the training data -- same cleaning, same
engineered features, same feature split -- except `drop_duplicate_rows()`: every row we are
asked about needs a prediction, so nothing is ever dropped (rows out = rows in).
"""
import pandas as pd

from src.data import load_data
from src.preprocessing import add_engineered_features, clean_dataset, split_features_target


def predict_new_rows(pipeline, config: dict) -> pd.DataFrame:
    """
    Reads `config["predict"]["path"]` (no target column), prepares it like the training data and
    returns one row per input row: the id column, the probability of class 1 and the 0/1
    prediction at `config["predict"]["threshold"]` (default 0.5). `pipeline` must be fitted.
    """
    predict_config = config["predict"]
    preprocessing_config = config["preprocessing"]
    id_column = config["diagnostics"].get("id_column")

    df_new = load_data(predict_config["path"])
    df_clean = clean_dataset(df_new, config["diagnostics"])         # row-preserving, no de-duplication
    if preprocessing_config.get("engineered_features", False):
        df_clean = add_engineered_features(df_clean)
    X_new, _, _ = split_features_target(df_clean, config["data"],
                                        preprocessing_config.get("mnar_indicator_sources", []))

    proba = pipeline.predict_proba(X_new)[:, 1]
    threshold = predict_config.get("threshold", 0.5)
    predictions = pd.DataFrame({
        id_column: df_new[id_column].values,
        "probability": proba.round(4),
        "prediction": (proba >= threshold).astype(int),
    })
    if len(predictions) != len(df_new):
        raise ValueError(f"{len(df_new)} rows in, {len(predictions)} predictions out: a row was dropped")
    return predictions
