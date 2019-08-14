# implemented by JunfengHu
# create time: 7/20/2019

import numpy as np
import torch
import config
from loss import linearizedlie_loss
import utils
from choose_dataset import DatasetChooser
from torch import autograd
from STLN import ST_HMR, ST_LSTM
import choose_dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    config = config.TrainConfig('Human', 'lie', 'all')
    # choose = DatasetChooser(config)
    # prediction_dataset, _ = choose(prediction=True)
    # prediction_loader = DataLoader(prediction_dataset, batch_size=config.batch_size, shuffle=True)

    choose = DatasetChooser(config)
    train_dataset, bone_length = choose(train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset, _ = choose(train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    prediction_dataset, bone_length = choose(prediction=True)
    prediction_loader = DataLoader(prediction_dataset, batch_size=config.batch_size, shuffle=True)



