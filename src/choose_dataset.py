# implemented by JunfengHu
# create time: 7/20/2019
import numpy as np
import torch
import scipy.io as sio
import load_data as loader


class DatasetChooser(object):

    def __init__(self, config):
        self.config = config
        self.dataset = config.dataset

    def choose_dataset(self, train=True):
        if self.dataset == 'Human':
            bone_length_path = './data/Human/Test/y_test_lie/directions' \
                               '_0_lie.mat'
            rawdata = sio.loadmat(bone_length_path)
            rawdata = rawdata[list(rawdata.keys())[3]]
            bone = self.cal_bone_length(rawdata)
            data = loader.HumanDataset(self.config, train=train)
            self.config.input_size = data[1]['encoder_inputs'].shape[1]
        elif self.dataset == 'Fish':
            pass
        elif self.dataset == 'Mouse':
            pass
        elif self.dataset == 'CSL':
            pass

        return data, bone

    def __call__(self, train=True):
        return self.choose_dataset(train)

    def cal_bone_length(self, rawdata):

        if self.config.datatype == 'lie':
            njoints = rawdata.shape[1]
            bone = np.zeros([njoints, 3])
            # 最后一个不需要，因为是第6个chain开始的下标，一共只有5个
            bone_skip = self.config.skip[0:-1]
            for i in range(njoints):
                if i in bone_skip:
                    continue
                else:
                    bone[i, 0] = round(rawdata[0, i, 3], 2)
        elif self.config.datatype == 'xyz':
            pass

        return bone

    # def set_config(self):
    #     if self.train:
    #         seq_length_out = self.config.output_window_size
    #         self.config.output_window_size
    #     else:
    #         seq_length_out = self.config.test_output_window
