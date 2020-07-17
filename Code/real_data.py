"""
Experiments on real data with categorical variables

"""
import pandas as pd
from sklearn import datasets
import joblib

mem = joblib.Memory(location='cache')

# %%
# Download the data

raw_data = dict()

openml_ids = {
    # https://www.openml.org/d/1189 BNG(auto_price)
    # No high-cardinality categories
    'BNG(auto_price)': 1189,
    ## https://www.openml.org/d/42160
    ## A few high-cardinality strings and some dates
    'la_crimes': 42160,
    # https://www.openml.org/d/42208 nyc-taxi-green-dec-2016
    # No high cardinality strings, a few datetimes
    # Skipping because I cannot get it to encode right
    #'nyc-taxi-green-dec-2016': 42208,
    # No high-cardinality categories
    # https://www.openml.org/d/41540
    'black_friday': 41540,
}

age_mapping = {
    '0-17': 15,
    '18-25': 22,
    '26-35': 30,
    '36-45': 40,
    '46-50': 48,
    '51-55': 53,
    '55+': 60
    }


for name, openml_id in openml_ids.items():
    X, y = mem.cache(datasets.fetch_openml)(
                                data_id=openml_id, return_X_y=True,
                                as_frame=True)
    raw_data[name] = X, y


# %%
# Encode the data to numerical matrices

clean_data = dict()

for name, (X, y) in raw_data.items():
    print('\nBefore encoding: % 20s: n=%i, d=%i'
          % (name, X.shape[0], X.shape[1]))
    if hasattr(X, 'columns'):
        print(list(X.columns))
        X = X.copy()
        if 'Age' in X.columns:
            X['Age'] = X['Age'].replace(age_mapping)
        columns_kept = []
        for col, dtype in zip(X.columns, X.dtypes):
            if col.endswith('datetime'):
                # Only useful for NYC taxi
                col_name = col[:-8]
                datetime = pd.to_datetime(X[col])
                X[col_name + 'year'] = datetime.dt.year
                X[col_name + 'weekday'] = datetime.dt.dayofweek
                X[col_name + 'yearday'] = datetime.dt.dayofyear
                X[col_name + 'time'] = datetime.dt.time
                columns_kept.extend([col_name + 'year',
                                     col_name + 'weekday',
                                     col_name + 'yearday',
                                     col_name + 'time'])
            elif dtype.kind in 'if':
                columns_kept.append(col)
            elif hasattr(dtype, 'categories'):
                if len(dtype.categories) < 30:
                    columns_kept.append(col)
            elif dtype.kind == 'O':
                if X[col].nunique() < 30:
                    columns_kept.append(col)
        X = X[columns_kept]

        X_array = pd.get_dummies(X).values
    else:
        X_array = X
    clean_data[name] = X_array, y
    print('After encoding: % 20s: n=%i, d=%i'
          % (name, X_array.shape[0], X_array.shape[1]))

# Save some memory
del raw_data


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_dataset(name):
    X, y = clean_data[name]
    n_axes_x = int(np.sqrt(X.shape[1]))
    n_axes_y = int(np.ceil(X.shape[1] / n_axes_x))

    corr = np.corrcoef(X.T)
    plt.matshow(corr, vmax=1, vmin=-1, cmap=plt.cm.RdYlBu_r)
    plt.colorbar()
    plt.title(name)
    plt.savefig(f'corr_{name}.png')

    fig, axes = plt.subplots(n_axes_x, n_axes_y, figsize=(12, 8))
    for i, (X_i, ax) in enumerate(zip(X.T, axes.ravel())):
        sns.kdeplot(X_i[:10000], y[:10000], ax=ax)
        ax.set_title('Feature %i' % i)
    plt.suptitle(name)
    plt.tight_layout(rect=[0, 0, 1, .97])
    plt.savefig(f'links_{name}.png')


# This can take a lot of time and RAM: have a big computer, and
# consider downsampling the data
#plot_dataset('BNG(auto_price)')


# %%
# some simple learning to check things are OK
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn import ensemble, model_selection

scores = dict()

for name, (X, y) in clean_data.items():
    n_samples = len(X)
    # Have a train set of at most 300000 data points
    test_size = max(int(.2 * n_samples), n_samples - 300000)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
        test_size=test_size)

    model = ensemble.HistGradientBoostingRegressor(verbose=1)

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print('Data: % 20s | score: %.3f' % (name, score))
    scores[name] = score

