# implemented by JunfengHu
# create time: 7/20/2019
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import copy


class FormatDataPre(object):

    def __init__(self):
        pass

    def __call__(self, x_test, y_test):

        dec_in_test = x_test[-1:, :]
        x_test = x_test[:-1, :]
        return {'x_test': x_test, 'dec_in_test': dec_in_test, 'y_test': y_test}


class FormatData(object):

    def __init__(self, config):
        self.config = config

    def __call__(self, sample, train):

        total_frames = self.config.input_window_size + self.config.output_window_size

        video_frames = sample.shape[0]
        idx = np.random.randint(25, video_frames-total_frames)

        data_seq = sample[idx:idx+total_frames, :]
        encoder_inputs = data_seq[:self.config.input_window_size-1, :]
        # 最后一个弃掉了,这里代码还可以精简
        if train:
            decoder_inputs = data_seq[self.config.input_window_size-1:
                                      self.config.input_window_size-1+self.config.output_window_size, :]
        else:
            decoder_inputs = data_seq[self.config.input_window_size - 1:self.config.input_window_size, :]
        decoder_outputs = data_seq[self.config.input_window_size:, :]
        return {'encoder_inputs': encoder_inputs, 'decoder_inputs': decoder_inputs, 'decoder_outputs': decoder_outputs}


class LieTsfm(object):

    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        rawdata = sample

        nframes = rawdata.shape[0]
        njoints = rawdata.shape[1]
        data = np.zeros([nframes, njoints+1, 3])
        # Reorganising the Lie algebra parameters to remove redundancy
        # The same as HMR
        data[:, 0, :] = rawdata[:, 0, 3:6]
        data[:, 1:, :] = rawdata[:, :, 0:3]
        # 这里wu shuang的代码写错了，除了skip数组的最后一个在原始data中应该索引不到以外
        # 所有都向后面移了一个
        #data = np.delete(data, [self.config.skip[:-1]+1], axis=1)
        data = np.delete(data, [self.config.skip], axis=1)
        data = np.around(data, 5)
        data = data.reshape(data.shape[0], -1)
        return data


class H36mDataset(Dataset):
    """
    This dataset only contains lie algebra data
    Part of the code is copied from: https://github.com/BII-wushuang/Lie-Group-Motion-Prediction
    """

    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        if train:
            subjects = [1, 6, 7, 8, 9, 11]
        else:
            subjects = [5]
        data_dir = './data/h3.6m/dataset'

        if config.filename == 'all':
            actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',
                 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']
        else:
            actions = [config.filename]

        set, complete_set = self.load_data(data_dir, subjects, actions)
        data_mean, data_std, dim_to_ignore, dim_to_use = self.normalization_stats(complete_set)

        if train:
            # Compute normalization stats
            data_mean, data_std, dim_to_ignore, dim_to_use = self.normalization_stats(complete_set)
            config.data_mean = data_mean
            config.data_std = data_std
            config.dim_to_ignore = dim_to_ignore
            config.dim_to_use = dim_to_use

            config.chain_idx = [np.array([0, 1, 2, 3, 4, 5]),
                                np.array([0, 6, 7, 8, 9, 10]),
                                np.array([0, 12, 13, 14, 15]),
                                np.array([13, 17, 18, 19, 22, 19, 21]),
                                np.array([13, 25, 26, 27, 30, 27, 29])]

        set = self.normalize_data(set, data_mean, data_std, dim_to_use)
        set_list = []
        for key in set.keys():
            set_list.append(set[key])

        self.data = set

    def load_data(self, data_dir, subjects, actions):
        """
           Copied from https://github.com/una-dinosauria/human-motion-prediction
        """
        train_data = {}
        complete_data = []
        for subj in subjects:
            for action in actions:
                for subact in [1, 2]:  # subactions
                    # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(data_dir, subj, action, subact)
                    action_sequence = self.readCSVasFloat(filename)

                    n, d = action_sequence.shape
                    even_list = range(0, n, 2)

                    train_data[(subj, action, subact, 'even')] = action_sequence[even_list, :]

                if len(complete_data) == 0:
                    complete_data = copy.deepcopy(action_sequence)
                else:
                    complete_data = np.append(complete_data, action_sequence, axis=0)

        return [train_data, complete_data]

    def readCSVasFloat(self, filename):
        """
        Copied from https://github.com/una-dinosauria/human-motion-prediction
        """
        return_array = []
        lines = open(filename).readlines()
        for line in lines:
            line = line.strip().split(',')
            if len(line) > 0:
                return_array.append(np.array([np.float32(x) for x in line]))
        return_array = np.array(return_array)
        return return_array

    def normalization_stats(self, complete_data):
        """
        Copied from https://github.com/una-dinosauria/human-motion-prediction
        """
        data_mean = np.mean(complete_data, axis=0)
        data_std = np.std(complete_data, axis=0)

        dimensions_to_ignore = []
        dimensions_to_use = []

        dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
        dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

        data_std[dimensions_to_ignore] = 1.0

        return [data_mean, data_std, dimensions_to_ignore, dimensions_to_use]

    def __getitem__(self, idx):

        sample = self.formatdata(self.data[idx])
        return sample

    def __len__(self):

        return len(self.data)


class HumanDataset(Dataset):
    """Dismissed due to some reasons"""

    def __init__(self, config, train=True):

        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        if config.datatype == 'lie':
            train_path = './data/Human/Train/train_lie/'
            tail = '_lie.mat'
        elif config.datatype == 'xyz':
            train_path = './data/Human/Train/train_lie/'
            tail = '_xyz.mat'
        if train:
            subjects = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']
        else:
            subjects = ['S5']

        if config.filename == 'all':
            actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',
                 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']
        else:
            actions = [config.filename]

        set = []
        for action in actions:
            for id in subjects:
                for i in range(2):
                    filename = train_path + id + '_' + action + '_' + str(i+1) + tail
                    rawdata = sio.loadmat(filename)
                    rawdata = rawdata[list(rawdata.keys())[3]]
                    set.append(rawdata)
        self.data = set

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            sample = self.lie_tsfm(self.data[idx])
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(sample)
        return sample


class FishDataset(Dataset):

    def __init__(self, config, train=True):

        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        if config.datatype == 'lie':
            train_path = './data/Fish/Train/train_lie/'
            tail = '_lie.mat'
        elif config.datatype == 'xyz':
            train_path = './data/Fish/Train/train_xyz/'
            tail = '_xyz.mat'
        if train:
            subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S8']
        else:
            subjects = ['S6']

        set = []
        for id in subjects:
            filename = train_path + id + tail
            rawdata = sio.loadmat(filename)
            rawdata = rawdata[list(rawdata.keys())[3]]
            set.append(rawdata)
        self.data = set

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            sample = self.lie_tsfm(self.data[idx])
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(sample, False)
        return sample

    def __len__(self):

        return len(self.data)


class MouseDataset(Dataset):

    def __init__(self, config, train):
        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        if config.datatype == 'lie':
            train_path = './data/Mouse/Train/train_lie/'
            tail = '_lie.mat'
        elif config.datatype == 'xyz':
            train_path = './data/Mouse/Train/train_xyz/'
            tail = '_xyz.mat'
        if train:
            subjects = ['S1',  'S3', 'S4']
        else:
            subjects = ['S2']

        set = []
        for id in subjects:
            filename = train_path + id + tail
            rawdata = sio.loadmat(filename)
            rawdata = rawdata[list(rawdata.keys())[3]]
            set.append(rawdata)
        self.data = set

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            sample = self.lie_tsfm(self.data[idx])
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(sample, self.train)
        return sample

    def __len__(self):

        return len(self.data)


class CSLDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, index: int):
        pass

    def __len__(self) :
        pass


class FishPredictionDataset(Dataset):

    def __init__(self, config):
        self.config = config
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatDataPre()
        if config.datatype == 'lie':
            x = []
            y = []
            for i in range(8):
                x_filename = './data/Fish/Test/x_test_lie/test_' + str(i) + '_lie.mat'
                y_filename = './data/Fish/Test/y_test_lie/test_' + str(i) + '_lie.mat'

                x_rawdata = sio.loadmat(x_filename)
                x_rawdata = x_rawdata[list(x_rawdata.keys())[3]]

                y_rawdata = sio.loadmat(y_filename)
                y_rawdata = y_rawdata[list(y_rawdata.keys())[3]]

                x.append(x_rawdata)
                y.append(y_rawdata)
            self.x = x
            self.y = y
        else:
            pass

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            x_sample = self.lie_tsfm(self.x[idx])
            y_sample = self.lie_tsfm(self.y[idx])
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(x_sample, y_sample)
        return sample


class MousePredictionDataset(Dataset):

    def __init__(self, config):

        if config.datatype == 'lie':
            x = []
            y = []
            for i in range(8):
                x_filename = './data/Mouse/Test/x_test_lie/test_' + str(i) + '_lie.mat'
                y_filename = './data/Mouse/Test/y_test_lie/test_' + str(i) + '_lie.mat'

                x_rawdata = sio.loadmat(x_filename)
                x_rawdata = x_rawdata[list(x_rawdata.keys())[3]]

                y_rawdata = sio.loadmat(y_filename)
                y_rawdata = y_rawdata[list(y_rawdata.keys())[3]]

                x.append(x_rawdata)
                y.append(y_rawdata)
            self.x = x
            self.y = y
        else:
            pass

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            x_sample = self.lie_tsfm(self.x[idx])
            y_sample = self.lie_tsfm(self.y[idx])
        elif self.config.datatype == 'xyz':
            pass
        return {'x': x_sample, 'y': y_sample}



