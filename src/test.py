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

    config = config.TrainConfig('Fish', 'lie', 'all')
    choose = DatasetChooser(config)
    prediction_dataset, _ = choose(prediction=True)
    prediction_loader = DataLoader(prediction_dataset, batch_size=config.batch_size, shuffle=True)

    # data = choose_dataset.DatasetChooser(config)
    # data, bone = data()
    # train_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    # for i, data in enumerate(train_loader, 0):
    #     print(data['encoder_inputs'].shape)
    #     break
    #
    # net = ST_HMR(config, True, bone.shape[0]-1)
    # #with autograd.detect_anomaly():
    # prediction = net(data['encoder_inputs'].float(), data['decoder_inputs'].float())
    # loss = linearizedlie_loss(prediction, data['decoder_outputs'].float(), bone)
    # loss.backward()
    # print(prediction.shape)
    # print(data['decoder_outputs'].float().shape)

    #st_lstm = STLSTM(config, True, 23)
    # hidden_states = torch.randn(8, 10, 23, 16)
    # cell_states = torch.randn(8, 10, 23, 16)
    # global_t_state = torch.randn(8,  23, 16)
    # global_s_state = torch.randn(8, 10, 16)
    # p = torch.randn(8, 10, 23, 16)
    #with autograd.detect_anomaly():
    # h, c = st_lstm(hidden_states, cell_states, global_t_state, global_s_state, p)
    # loss = torch.sum(h)
    # loss.backward()
    # print(sum(p.numel() for p in net.parameters()))

