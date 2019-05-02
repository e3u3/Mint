import numpy as np


def alloc_idxs(batch_size, idxs, data_size):

    res = list()
    start = 0
    while start < data_size:
        end = min(start+batch_size, data_size)
        tmp = idxs[start:end]
        res.append(tmp)
        start = end
    
    return res

def get_random_idxs(batch_size, data_size):

    idxs = np.arange(data_size)
    np.random.shuffle(idxs)
    
    return alloc_idxs(batch_size, idxs, data_size)

def get_deter_idxs(batch_size, data_size):

    idxs = np.arange(data_size)

    return alloc_idxs(batch_size, idxs, data_size)

def next_batch(data_label, batch_idx, random_idxs):

    idxs = random_idxs[(batch_idx) % len(random_idxs)]
    return data_label[0][idxs], data_label[1][idxs]

def correct(output, target):

    pred = np.argmax(output, axis=1)
    return np.sum(pred == target)

def to_categorical(y, num_classes=None):

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical