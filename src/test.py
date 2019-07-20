# implemented by JunfengHu
# create time: 7/20/2019

import numpy as np
import torch
import config
import utils
import choose_dataset
if __name__ == '__main__':

    config = config.TrainConfig('Human', 'lie', 'all')
    data = choose_dataset.DatasetChooser(config, True)
    data, bone = data()
    a = data[1]['data']