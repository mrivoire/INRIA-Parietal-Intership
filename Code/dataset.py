import pandas as pd
from sklearn import datasets
import joblib
import faulthandler


def load_auto_prices():
    """Load Auto Prices Dataset

    https://www.openml.org/d/1189 BNG(auto_price)
    No high-cardinality categories
    """
    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    X, y = mem.cache(datasets.fetch_openml)(
        data_id=1189, return_X_y=True, as_frame=True)
    return X, y


def load_lacrimes():
    """Load Crimes Dataset

    https://www.openml.org/d/42160
    A few high-cardinality strings and some dates
    """
    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    X, y = mem.cache(datasets.fetch_openml)(
        data_id=42160, return_X_y=True, as_frame=True)

    X['Date_Reported'] = pd.to_datetime(X['Date_Reported'])
    X['Date_Occurred'] = pd.to_datetime(X['Date_Occurred'])

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')

    return X, y


def load_black_friday():
    """Load Black Friday Dataset

    No high-cardinality categories
    https://www.openml.org/d/41540
    """
    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    X, y = mem.cache(datasets.fetch_openml)(
        data_id=41540, return_X_y=True, as_frame=True)

    age_mapping = {
        '0-17': 15,
        '18-25': 22,
        '26-35': 30,
        '36-45': 40,
        '46-50': 48,
        '51-55': 53,
        '55+': 60
    }

    X['Age'] = X['Age'].replace(age_mapping)

    return X, y


def load_nyc_taxi():
    """Load NYC Taxi Dataset

    https://www.openml.org/d/42208 nyc-taxi-green-dec-2016
    No high cardinality strings, a few datetimes
    """
    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    X, y = mem.cache(datasets.fetch_openml)(
        data_id=42208, return_X_y=True, as_frame=True)

    X['lpep_pickup_datetime'] = pd.to_datetime(X['lpep_pickup_datetime'])
    X['lpep_dropoff_datetime'] = pd.to_datetime(X['lpep_dropoff_datetime'])

    return X, y


def main():
    # Load Auto Prices Dataset
    X_auto_prices, y_auto_prices = load_auto_prices()
    print("X = ", X_auto_prices.dtypes)

    # Load LA Crimes Dataset
    X_lacrimes, y_lacrimes = load_lacrimes()
    print("X = ", X_lacrimes.dtypes)

    # Load Black Friday Dataset
    X_black_friday, y_black_friday = load_black_friday()
    print("X_black_friday = ", X_black_friday.dtypes)

    # Load NYC Taxi Dataset
    X_nyc_taxi, y_nyc_taxi = load_nyc_taxi()
    print("X_nyc_taxi Types = ", X_nyc_taxi.dtypes)


if __name__ == "__main__":
    main()
