from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
)
from xgboost import XGBClassifier


# Binary classification
def predict_long_trip(df):
    # Add binary label
    df["is_long_trip"] = (df["trip_distance"] > 10).astype(int)

    features = ["passenger_count", "fare_amount", "tip_amount", "pickup_hour"]
    target = "is_long_trip"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    print(classification_report(y_test, y_pred))
