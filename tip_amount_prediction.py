from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score,
)
from sklearn.ensemble import RandomForestRegressor


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

    # Try Random Forest Regressor
    rf = RandomForestRegressor()
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    # Metrics
    print("Random Forest:")
    print("MAE:", mean_absolute_error(y_test, y_pred_rf))
    print("RMSE:", mean_squared_error(y_test, y_pred_rf))
    print("R²:", r2_score(y_test, y_pred_rf))
