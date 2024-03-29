import numpy as np
import datetime


def _vectorize(description, vectorizer):
    arr = description.split(" ")
    r = None
    for a in arr:
        try:
            if r is None:
                r = np.array([vectorizer[a]])
            else:
                r = np.vstack([r, vectorizer[a]])
        except KeyError:
            pass
    if r is None:
        return np.array([])
    return r


def _padArr(arr, arr_size, tot_size):
    if len(arr) == 0:
        return np.zeros((tot_size, arr_size))
    for i in range(len(arr), tot_size):
        arr = np.vstack([arr, np.zeros(arr_size)])
    return arr


def _padSer(ser, arr_size):
    tot_size = ser.apply(len).max()
    return ser.apply(lambda arr: _padArr(arr, arr_size, tot_size))


def _max_open_contracts(ser):
    if len(ser) == 1:
        return 1
    dates_to_ck = list(ser.apply(lambda x: x[0])) + list(
        ser.apply(lambda x: x[1] + datetime.timedelta(days=1))
    )
    return max(
        [
            ((ser.apply(lambda x: x[0]) <= d) & (ser.apply(lambda x: x[1]) >= d)).sum()
            for d in dates_to_ck
        ]
    )
