
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from datasets.loader.dataset_loader.dataset_loader import DatasetLoader


class YellowTaxiDatasetLoader(DatasetLoader):

    def load(self, dataset_type, **kwargs):
        interval = 10 # Minutes
        window = 24 * 60 // interval # Number of timesteps

        # Load data
        path = "datasets/yellow-taxi"

        if (dataset_type == "train"):
            files = [f"{path}/yellow_tripdata_2016-0{i}.csv" for i in range(1,4)]
            df = pd.concat([pd.read_csv(file) for file in files], ignore_index = True)
        else:
            file = f"{path}/yellow_tripdata_2015-01.csv"
            df = pd.read_csv(file)
        
        # Prepare data
        df = self.__clean(df)
        df = self.__transform(df, interval)

        if (dataset_type == "train"):
            # Scale attributes
            self.scaler = MinMaxScaler()
            df = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)

            # Split into training and validation sets (80%-20%)
            train_size = int(len(df) * 0.8)
            train, val = df[:train_size], df[train_size:]

            # Shape sets into windows
            trainX, trainY = self.__windows(np.array(train), window)
            valX, valY = self.__windows(np.array(val), window)
            
            return trainX, valX, trainY, valY
        
        else:
            df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)
            testX, testY = self.__windows(np.array(df), window)

            return testX, testY
            


    def __clean(self, df):
        # Keep only relevant atributes
        df = df[[
            'passenger_count','trip_distance','pickup_longitude','pickup_latitude',
            'dropoff_longitude','dropoff_latitude', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'total_amount'
        ]]

        # Remove passenger count outliers (more than 6 passengers or negative)
        df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

        # New York boundaries
        lat_min, lat_max = 40.5774, 40.9176
        lon_min, lon_max = -74.15, -73.7004

        df = df[
            (df["pickup_latitude"].between(lat_min, lat_max)) &
            (df["pickup_longitude"].between(lon_min, lon_max)) &
            (df["dropoff_latitude"].between(lat_min, lat_max)) &
            (df["dropoff_longitude"].between(lon_min, lon_max))
        ]

        # Trip time (minutes)
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

        df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

        # Remove trips with negative duration or over 12 hours (limit)
        df = df[(df["trip_duration"] > 0) & (df["trip_duration"] < 12 * 60)]

        # Remove distance outliers
        df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 23)]

        # Remove speed outliers
        df["speed"] = 60 * df["trip_distance"] / df["trip_duration"] # miles/h
        df = df[(df["speed"] > 0) & (df["speed"] < 45.31)]

        # Remove fare outliers
        df = df[(df["total_amount"] > 0) & (df["total_amount"] < 1000)]

        # Remove attributes not used in training/testing
        df.drop(columns=["tpep_dropoff_datetime",'pickup_longitude','pickup_latitude','dropoff_longitude',
                        'dropoff_latitude'], inplace = True)

        return df


    def __transform(self, df, interval):
        # Set pickup timestamp as index
        df.set_index("tpep_pickup_datetime", inplace=True)

        # Compute interval aggregations
        df = df.resample(f"{interval}min").agg({
            "passenger_count": "sum",
            "trip_distance": ["sum", "mean"],
            "trip_duration": ["sum", "mean"],
            "total_amount": "sum",
            "speed": ["mean", "count"]  # count is for the number of trips
        })

        df.columns = ["passenger_sum", "distance_sum", "distance_mean", "duration_sum",
                    "duration_mean", "total_sum", "speed_mean", "trip_count"]
        
        # In intervals with no trips, the means are Nan (0/0), so they are filled with zeros
        df.fillna(0, inplace=True)

        # Add time info
        df["day_of_year"] = df.index.dayofyear
        df["weekday"] = df.index.weekday
        df["hour"] = df.index.hour

        # Move trip count to the end
        df["trip_count"] = df.pop("trip_count")

        return df


    def __windows(self, data, timesteps):
        x, y = [], []
        for i in range(len(data) - timesteps):
            x.append(data[i:i+timesteps])  # Input window
            y.append(data[i+timesteps, -1])  # Next trip count
        return np.array(x), np.array(y)