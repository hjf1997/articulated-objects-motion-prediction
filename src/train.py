# implemented by JunfengHu
# created time: 7/20/2019

import torch
from torch.utils.data import DataLoader
import numpy as np
import utils
from choose_dataset import DatasetChooser
from loss import linearizedlie_loss
import utils
import scipy.io as sio
import torch.optim as optim
import re
import config
from STLN import ST_HMR

def train(config):

    print('Start Training the Model!')

    # generate data loader
    choose = DatasetChooser(config)
    train_dataset, bone_length = choose(train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_dataset, _ = choose(train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used'.format(device))
    torch.cuda.manual_seed(971103)
    net = ST_HMR(config, True, bone_length.shape[0]-1)
    net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(str(torch.cuda.device_count())))
        net = torch.nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    for epoch in range(config.max_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            encoder_inputs = data['encoder_inputs'].float().to(device)
            decoder_inputs = data['decoder_inputs'].float().to(device)
            decoder_outputs = data['decoder_outputs'].float().to(device)
            prediction = net(encoder_inputs, decoder_inputs)
            loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
            running_loss += loss
            net.zero_grad()
            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
        print("epoch:{}, train_loss:{}".format(str(epoch + 1), str((running_loss / (i + 1)).item())))

if __name__ == '__main__':

    config = config.TrainConfig('Fish', 'lie', 'all')
    train(config)