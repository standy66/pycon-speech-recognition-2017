import numpy as np
import tensorflow as tf
import os
from itertools import zip_longest
from scipy.io import wavfile
from scipy.signal import spectrogram
from glob import iglob


def extract_feats(sample_rate, data, window_size_msec=0.025, window_offset_msec=0.01):
    """Extract spectrogram features from audio data."""
    offset_frames = int(sample_rate * window_offset_msec)
    window_frames = int(sample_rate * window_size_msec)
    fs, ts, Sxx = spectrogram(data, sample_rate, nperseg=window_frames, noverlap=window_frames-offset_frames)
    Sxx = np.log(1e-9 + Sxx)
    return fs, ts, Sxx.T


def normalize(data):
    """Normalize audio data so that samples are in range [-1.0, 1.0]."""
    if data.dtype == np.uint8:
        return (data.astype(np.float) - 128) / 128
    elif data.dtype == np.int16:
        return (data.astype(np.float)) / 2 ** 15
    else:
        raise ValueError("Unknown dtype")
    
    
def read_wav(path):
    """Read wav file referenced by path using scipy."""
    sample_rate, data = wavfile.read(path)
    data = normalize(data)
    if len(data.shape) == 1:
        data = np.reshape(data, data.shape + (1,))
    assert(len(data.shape) == 2)
    return sample_rate, data


def read_dataset(path):
    """Read free-spoken-digit-dataset at path."""
    X = []
    y = []
    for filename in iglob("{path}/*.wav".format(path=path)):
        digit, speaker, _ = os.path.basename(filename).split("_")
        _, data = read_wav(filename)
        # NOTE: may try to pick a better constant to account for different loudness.
        # if speaker == "nicolas":
        #     data = data * 3
        for channel in range(data.shape[1]):
            X.append(data[:, channel])
            y.append(int(digit))
    return X, y


def list_2d_to_sparse(list_of_lists):
    """Convert python list of lists to a [tf.SparseTensorValue](https://www.tensorflow.org/api_docs/python/tf/SparseTensorValue).

    Args:
        list_of_lists: list of lists to convert.

    Returns:
        tf.SparseTensorValue which is a namedtuple (indices, values, shape) where:

            * indices is a 2-d numpy array with shape (sum_all, 2) where sum_all is a
            sum over i of len(l[i])

            * values is a 1-d numpy array with shape (sum_all, )

            * shape = np.array([len(l), max_all]) where max_all is a max over i of
            len(l[i])

        Also, the following is true: for all i values[i] ==
        list_of_lists[indices[i][0]][indices[i][1]]

    """
    indices, values = [], []
    for i, sublist in enumerate(list_of_lists):
        for j, value in enumerate(sublist):
            indices.append([i, j])
            values.append(value)
    dense_shape = [len(list_of_lists), max(map(len, list_of_lists))]
    return tf.SparseTensorValue(indices=np.array(indices),
                                values=np.array(values),
                                dense_shape=np.array(dense_shape))


def batch(X, y, batch_size):
    num_features = X[0].shape[1]
    n = len(X)
    perm = np.random.permutation(n)
    for batch_ind in np.resize(perm, (n // batch_size, batch_size)):
        X_batch, y_batch = [X[i] for i in batch_ind], [y[i] for i in batch_ind]
        sequence_lengths = list(map(len, X_batch))
        X_batch_padded = np.array(list(zip_longest(*X_batch, fillvalue=np.zeros(num_features)))).transpose([1, 0, 2])
        yield X_batch_padded, sequence_lengths, list_2d_to_sparse(y_batch), y_batch
        

def decode(d, mapping):
    """Decode."""
    shape = d.dense_shape
    batch_size = shape[0]
    ans = np.zeros(shape=shape, dtype=int)
    seq_lengths = np.zeros(shape=(batch_size, ), dtype=np.int)
    for ind, val in zip(d.indices, d.values):
        ans[ind[0], ind[1]] = val
        seq_lengths[ind[0]] = max(seq_lengths[ind[0]], ind[1] + 1)
    ret = []
    for i in range(batch_size):
        ret.append("".join(map(lambda s: mapping[s], ans[i, :seq_lengths[i]])))
    return ret


class RandomSequencer:
    def __init__(self, X, y, sample_rate):
        self.X = X
        self.y = y
        self.sample_rate
        self.dataset_size = len(X)
        
    def generate(self, length, silence_from=0.5, silence_to=1.0, noise_db=-10):
        seq_parts = []
        y_seq_parts = []
        seq_parts.append(np.zeros(int(self.sample_rate * np.random.uniform(silence_from, silence_to))))
        for i in range(length):
            selected_id = np.random.choice(self.dataset_size)
            seq_parts.append(self.X[selected_id])
            seq_parts.append(np.zeros(int(self.sample_rate * np.random.uniform(silence_from, silence_to))))
            y_seq_parts.append(self.y[selected_id])
        audio = np.concatenate(seq_parts)
        signal_rmse = np.sqrt(np.var(audio))
        audio += np.random.normal(scale = 10 ** (noise_db / 20) * signal_rmse, size=audio.shape)
        return audio, y_seq_parts
    