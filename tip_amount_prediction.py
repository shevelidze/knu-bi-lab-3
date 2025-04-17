from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def tip_amount_prediction(df):
    # Features and target
    features = [
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "improvement_surcharge",
        "tolls_amount",
        "congestion_surcharge",
        "airport_fee",
        "passenger_count",
        "payment_type",
    ]
    target = "tip_amount"

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "Random Forest": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        print(f"\n{name}")
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("RMSE:", mean_squared_error(y_test, y_pred))  # RMSE
        print("R²:", r2_score(y_test, y_pred))
