import numpy as np
import pandas as pd
from sklearn import preprocessing

# Global parameters for the data
DATE_COL = 0
SYMBOL_COL = 1
OPEN_COL = 2
CLOSE_COL = 3
LOW_COL = 4
HIGH_COL = 5
VOLUME_COL = 6


def data_split(x_train, y_train, worker_number):
    """
    Split data between workers.

    Args:
        x_train (ndarray): Train data input.
        y_train (ndarray): Train data output.
        worker_number (int): Amount of workers.

    Returns:
        tuple: The data split as a tuple of lists of x and y values.

    """
    split_step = int(len(x_train) / worker_number)
    x_vec = []
    y_vec = []
    for i in range(worker_number):
        x_vec.append(x_train[(i * split_step):((i + 1) * split_step)])
        y_vec.append(y_train[(i * split_step):((i + 1) * split_step)])
    return x_vec, y_vec


def load_data2(data_path, history_window_size=20, data_ratio=0.85):
    """
    Load million song data and split it.

    Args:
        data_path (str): The path where the million song data is located.
        history_window_size (int): The size of the history window.
        data_ratio (float): The ratio between the training and validation data.

    Returns:
        tuple: The data split as a tuple of arrays of x and y values.

    """
    df = pd.read_csv(data_path)
    d = dict([(y, x + 1) for x, y in enumerate(sorted(set(df['symbol'])))])
    x = []
    y = []
    dates = []
    for symbol_id, symbol in enumerate(d):
        local_symbol = np.array(df[df['symbol'] == symbol])

        X = np.array(
            local_symbol[history_window_size:, [OPEN_COL, LOW_COL, HIGH_COL, VOLUME_COL]],
            dtype='float'
        )
        X = np.concatenate((X, (symbol_id + 1) * np.ones((len(X), 1)), np.ones((len(X), 1))), axis=1)
        num_orig_cols = X.shape[1]
        Y = np.array(
            local_symbol[history_window_size:, CLOSE_COL],
            dtype='float'
        )
        X = np.concatenate(
            (X, np.zeros((len(X), history_window_size))),
            axis=1
        )
        for row in range(len(X)):
            for day in range(1, history_window_size + 1):
                col_offset = num_orig_cols - 1 + day
                row_offset = history_window_size + row - day
                X[row, col_offset] = local_symbol[row_offset, CLOSE_COL]

        assert X.shape[1] == (history_window_size + num_orig_cols)

        dates_array = np.array(
            local_symbol[history_window_size:, [DATE_COL]],
            dtype='datetime64'
        )
        x.append(X)
        y.append(Y)
        dates.append(dates_array)

    x = np.concatenate(x, axis=0)
    x = preprocessing.normalize(x)
    y = np.concatenate(y, axis=0)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    x = x[idx, :]
    y = y[idx]
    x = np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)
    y = (y - y.min()) / (y.max() - y.min())

    x_train = x[:int(x.shape[0] * data_ratio), :]  # Splitting x_train data.
    y_train = y[:int(x.shape[0] * data_ratio)]  # Splitting y_train data.
    x_val = x[int(x.shape[0] * data_ratio):, :]  # Splitting x_val data.
    y_val = y[int(x.shape[0] * data_ratio):]  # Splitting y_val data.

    return x_train, y_train, x_val, y_val


def load_data(data_path, data_ratio=0.85):
    """
    Load million song data and split it.

    Args:
        data_path (str): The path where the million song data is located.
        data_ratio (float): The ratio between the training and validation data.

    Returns:
        tuple: The data split as a tuple of arrays of x and y values.

    """
    with open(data_path, 'r') as file:
        lines = file.readlines()
    data = np.array([np.array(line.split(',')).astype('float64') for line in lines])
    y = data[:, 0]
    x = data[:, 1:]
    x = x / np.linalg.norm(x, 1, axis=1)[:, None]
    x = np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)
    y = (y - y.min()) / (y.max() - y.min())
    x_train = x[:int(x.shape[0] * data_ratio), :]
    y_train = y[:int(x.shape[0] * data_ratio)]
    x_val = x[int(x.shape[0] * data_ratio):, :]
    y_val = y[int(x.shape[0] * data_ratio):]
    return x_train, y_train, x_val, y_val
