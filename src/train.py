# implemented by JunfengHu
# created time: 7/20/2019

import torch
from torch.utils.data import DataLoader
import numpy as np
from utils import Progbar
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
    train_loader = DataLoader(train_dataset, batch_size=config.file_batch, shuffle=True)
    test_dataset, _ = choose(train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.file_batch, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used'.format(device))
    torch.cuda.manual_seed(971103)
    net = ST_HMR(config, True, bone_length.shape[0]-1)
    net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(str(torch.cuda.device_count())))
        net = torch.nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(config.max_epoch):
        prog = Progbar(target=config.training_size)
        prog_valid = Progbar(target=config.validation_size)

        # Train
        for it in range(config.training_size):
            for j in range(config.data_batch):
                for i, data in enumerate(train_loader, 0):
                    if j == 0 and i == 0:
                        encoder_inputs = data['encoder_inputs'].float().to(device)
                        decoder_inputs = data['decoder_inputs'].float().to(device)
                        decoder_outputs = data['decoder_outputs'].float().to(device)
                    else:
                        encoder_inputs = torch.cat([data['encoder_inputs'].float().to(device), encoder_inputs], dim=0)
                        decoder_inputs = torch.cat([data['decoder_inputs'].float().to(device), decoder_inputs], dim=0)
                        decoder_outputs = torch.cat([data['decoder_outputs'].float().to(device), decoder_outputs], dim=0)
            prediction = net(encoder_inputs, decoder_inputs)
            loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
            prog.update(it+1, [("Training Loss", loss)])
            net.zero_grad()
            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        # Test
        for it in range(config.validation_size):
            for j in range(config.data_batch):
                for i, data in enumerate(test_loader, 0):
                    if j == 0 and i == 0:
                        encoder_inputs = data['encoder_inputs'].float().to(device)
                        decoder_inputs = data['decoder_inputs'].float().to(device)
                        decoder_outputs = data['decoder_outputs'].float().to(device)
                    else:
                        encoder_inputs = torch.cat([data['encoder_inputs'].float().to(device), encoder_inputs], dim=0)
                        decoder_inputs = torch.cat([data['decoder_inputs'].float().to(device), decoder_inputs], dim=0)
                        decoder_outputs = torch.cat([data['decoder_outputs'].float().to(device), decoder_outputs], dim=0)
            prediction = net(encoder_inputs, decoder_inputs)
            loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
            prog_valid.update(it+1, [("Training Loss", loss)])

if __name__ == '__main__':

    config = config.TrainConfig('Fish', 'lie', 'all')
    train(config)