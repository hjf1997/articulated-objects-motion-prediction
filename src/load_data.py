# implemented by JunfengHu
# create time: 7/20/2019
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio


class FormatData(object):

    def __init__(self, config):
        self.config = config

    def __call__(self, sample):

        total_frames = self.config.input_window_size + self.config.output_window_size

        #encoder_inputs = np.zeros((self.config.input_window_size - 1, self.config.input_size), dtype=float)
        #decoder_inputs = np.zeros((self.config.output_window_size, self.config.input_size), dtype=float)
        #decoder_outputs = np.zeros((self.config.output_window_size, self.config.input_size), dtype=float)

        video_frames = sample.shape[0]
        idx = np.random.randint(25, video_frames-total_frames)

        data_seq = sample[idx:idx+total_frames, :]
        encoder_inputs = data_seq[:self.config.input_window_size-1, :]
        # 最后一个弃掉了,这里代码还可以精简
        decoder_inputs = data_seq[self.config.input_window_size-1:
                                  self.config.input_window_size-1+self.config.output_window_size, :]
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


class HumanDataset(Dataset):

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
        sample = self.formatdata(sample)
        return sample

    def __len__(self):

        return len(self.data)


class MouseDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, index: int):
        pass

    def __len__(self) :
        pass


class CSLDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, index: int):
        pass

    def __len__(self) :
        pass


