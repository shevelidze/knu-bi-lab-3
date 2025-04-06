import pandas as pd

FILE_PATH = "light-data.csv"


def load_data_frame():
    df = pd.read_csv(FILE_PATH)

    # Convert datetime fields
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Compute trip duration in minutes
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # Filter out unrealistic values
    df = df[
        (df["trip_distance"] > 0)
        & (df["trip_distance"] < 500)
        & (df["fare_amount"] > 0)
        & (df["trip_duration"] < 120)
    ]

    # Fill NA
    df.fillna(0, inplace=True)

    # Add pickup hour
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

    return df
