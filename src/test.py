# implemented by JunfengHu
# create time: 7/20/2019

import numpy as np
import torch
import config
import utils
import choose_dataset
if __name__ == '__main__':

    config = config.TrainConfig('Human', 'lie', 'all')
    data = choose_dataset.DatasetChooser(config)
    data, bone = data()
    a = data[1]['encoder_inputs'].shape
    b = data[1]['decoder_inputs'].shape
    c = data[1]['decoder_outputs'].shape
    print(a, b, c)