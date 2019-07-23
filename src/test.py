# implemented by JunfengHu
# create time: 7/20/2019

import numpy as np
import torch
import config
import utils
from STLN import STLN
import choose_dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    config = config.TrainConfig('Human', 'lie', 'all')
    data = choose_dataset.DatasetChooser(config)
    data, bone = data()
    train_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    for i, data in enumerate(train_loader, 0):
        print(data['encoder_inputs'].shape)
        break

    net = STLN(config)
    final_hidden_states, final_cell_states, final_global_t_states, final_global_s_states = net(data['encoder_inputs'].float(), None)