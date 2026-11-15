"""Train the used-car price model from Car_data.csv."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Car_data.csv"
MODEL_PATH = ROOT / "car_prediction_model.pkl"


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    # Names in this dataset start with the make; the remaining text identifies
    # the model/trim and is kept intact so the app can offer matching choices.
    data["Brand"] = data["name"].str.split().str[0].str.strip()
    data["Model"] = data["name"].str.split(n=1).str[1].fillna("Standard")
    owner_map = {
        "First Owner": 0,
        "Second Owner": 1,
        "Third Owner": 2,
        "Fourth & Above Owner": 3,
        "Test Drive Car": 0,
    }
    data["Owner"] = data["owner"].map(owner_map)
    data["selling_price_lakhs"] = data["selling_price"] / 100000
    return data


def main() -> None:
    data = prepare_data(pd.read_csv(DATA_PATH))
    categorical = ["Brand", "Model", "fuel", "seller_type", "transmission"]
    numeric = ["year", "km_driven", "Owner"]
    features = categorical + numeric
    data = data.dropna(subset=features + ["selling_price_lakhs"])

    preprocessor = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numbers", "passthrough", numeric),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
            )),
        ]
    )
    model.fit(data[features], data["selling_price_lakhs"])
    joblib.dump(model, MODEL_PATH)
    print(f"Trained on {len(data):,} listings; saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
