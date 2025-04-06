from load_data_frame import load_data_frame

from tip_amount_prediction import tip_amount_prediction
from payment_type_classification import payment_data_type_classification
from trip_duration_clustering import trip_duration_clustering
from predict_long_trip import predict_long_trip

df = load_data_frame()
print("Data loaded successfully")
print("Processing data...")

print("\n1. Tip amount prediction")
tip_amount_prediction(df)

print("\n2. Payment type classification")
payment_data_type_classification(df)

print("\n3. Trip duration clustering")
trip_duration_clustering(df)

print("\n4. Long trip prediction")
predict_long_trip(df)
