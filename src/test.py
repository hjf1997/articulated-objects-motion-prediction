# implemented by JunfengHu
# create time: 7/20/2019

import numpy as np
import torch
import config
import utils
from torch import autograd
from STLN import STLN, STLSTM
import choose_dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    config = config.TrainConfig('Human', 'lie', 'all')
    # data = choose_dataset.DatasetChooser(config)
    # data, bone = data()
    # train_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    # for i, data in enumerate(train_loader, 0):
    #     print(data['encoder_inputs'].shape)
    #     break
    #
    # net = STLN(config, True, 23)
    # with autograd.detect_anomaly():
    #     final_hidden_states, final_cell_states, final_global_t_states, final_global_s_states = net(data['encoder_inputs'].float(), None)
    #     loss = torch.sum(final_hidden_states[-1])
    #     loss.backward()

    st_lstm = STLSTM(config, True, 23)
    hidden_states = torch.randn(8, 10, 23, 16)
    cell_states = torch.randn(8, 10, 23, 16)
    global_t_state = torch.randn(8,  23, 16)
    global_s_state = torch.randn(8, 10, 16)
    p = torch.randn(8, 10, 23, 16)
    with autograd.detect_anomaly():
        h, c = st_lstm(hidden_states, cell_states, global_t_state, global_s_state, p)
        loss = torch.sum(h)
    print(sum(p.numel() for p in st_lstm.parameters()))
    #loss.backward()
