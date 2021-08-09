import pickle
import numpy as np


def load_seq_dict(p, rescale=True):
    data_dict = pickle.load(open(p, 'rb'))

    if rescale:
        for k, v in data_dict.items():
            nums = np.array(v[0])[:v[2], :2]
            min_x, min_y = nums.min(axis=0)
            max_x, max_y = nums.max(axis=0)
            # mean_x, mean_y = nums.mean(axis=0)
            rangs_x, range_y = max_x - min_x, max_y - min_y

            v[0][:v[2], :2] = (v[0][:v[2], :2] - (min_x, min_y)) * 256. / max(rangs_x, range_y)
            data_dict[k] = v

    return data_dict


def load_quant_dict(p):
    data_dict = pickle.load(open(p, 'rb'))

    for k, v in data_dict.items():
        if len(v) < 150:
            v = np.concatenate((v, [1000] * (150 - len(v))))
            data_dict[k] = v

    return data_dict
