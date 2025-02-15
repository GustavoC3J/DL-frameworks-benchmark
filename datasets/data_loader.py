
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader():

    def __init__(self, model_type, seed):
        self.model_type = model_type
        self.seed = seed

    def load_data(self, dataset_type):
        if (self.model_type == "mlp"):
            return self.__load_fashion_mnist(dataset_type)
        
        elif (self.model_type == "cnn"):
            return self.__load_cifar_100(dataset_type)
        
        elif (self.model_type == "lstm"):
            return self.__load_yellow_taxi(dataset_type)


    def __load_fashion_mnist(self, dataset_type):

        if (dataset_type == "train"):
            # Load the training dataset
            data = pd.read_csv('datasets/fashion-mnist/fashion-mnist_train.csv')

            # Extract labels and pixel values
            labels = data['label']
            images = data.drop('label', axis=1).values

            # Scale the images between 0 and 1
            images = images / 255.0

            # Split into training and validation sets (80%-20%)
            # Returns trainX, validX, trainY, validY 
            return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=self.seed)

        else:
            # Load the test dataset
            test = pd.read_csv('datasets/fashion-mnist/fashion-mnist_test.csv')

            # Extract labels and pixel values
            testY = test['label']
            testX = test.drop('label', axis=1).values / 255.0

            return (testX, testY)
        

    def __load_cifar_100(self, dataset_type):
        path = f"datasets/cifar-100/{dataset_type}"

        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        def shape_images(img):
            num_images = img.shape[0]
            
            # Split into three channels
            images = img.reshape(num_images, 3, 1024)
            
            # Reshape to image width and height
            images = images.reshape(num_images, 3, 32, 32)
            
            # Change order of channels (num_imagenes, 32, 32, 3)
            images = images.transpose(0, 2, 3, 1)

            return images

        # Load data
        dataset = unpickle(path)
        images = dataset[b'data']
        labels = np.array(dataset[b'coarse_labels'])

        # Prepare data
        images = shape_images(images)
        images = images / 255.0 # Scale the images between 0 and 1

        if (dataset_type == "train"):
            # Split into training and validation sets (80%-20%)
            # Returns trainX, validX, trainY, validY 
            return train_test_split(images, labels, test_size = 0.2, stratify = labels, random_state = self.seed)
        else:
            return (images, labels)
        

    def __load_yellow_taxi(self, dataset_type):

        interval = 10 # Minutes
        window = 48 * 60 // interval # Number of timesteps

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
            # Scale attributes between 0 and 1
            self.scaler = StandardScaler()
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

            return self.__windows(np.array(df), window) # testX, testY


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

