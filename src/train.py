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
import os
from STLN import ST_HMR


def train(config):

    print('Start Training the Model!')

    # generate data loader
    choose = DatasetChooser(config)
    train_dataset, bone_length = choose(train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset, _ = choose(train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used'.format(device))
    torch.cuda.manual_seed(971103)
    net = ST_HMR(config, False, bone_length.shape[0]-1)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.st_lstm.parameters())))

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(str(torch.cuda.device_count())))
        net = torch.nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Save model
    checkpoint_dir = './model/'
    if not (os.path.exists(checkpoint_dir)):
        os.makedirs(checkpoint_dir)

    best_val_loss = float('inf')

    for epoch in range(config.max_epoch):
        print("At epoch:{}".format(str(epoch + 1)))
        prog = Progbar(target=config.training_size)
        prog_valid = Progbar(target=config.validation_size)

        # Train
        for it in range(config.training_size):
            run_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                encoder_inputs = data['encoder_inputs'].float().to(device)
                decoder_inputs = data['decoder_inputs'].float().to(device)
                decoder_outputs = data['decoder_outputs'].float().to(device)
                # else:
                #     encoder_inputs = torch.cat([data['encoder_inputs'].float().to(device), encoder_inputs], dim=0)
                #     decoder_inputs = torch.cat([data['decoder_inputs'].float().to(device), decoder_inputs], dim=0)
                #     decoder_outputs = torch.cat([data['decoder_outputs'].float().to(device), decoder_outputs], dim=0)
                prediction = net(encoder_inputs, decoder_inputs, True)
                loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
                run_loss += loss.item()
                net.zero_grad()
                loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
            prog.update(it + 1, [("Training Loss", run_loss/(i+1))])

        # Test
        test_loss = 0.0
        for it in range(config.validation_size):
                run_loss = 0.0
                for i, data in enumerate(test_loader, 0):
                    encoder_inputs = data['encoder_inputs'].float().to(device)
                    decoder_inputs = data['decoder_inputs'].float().to(device)
                    decoder_outputs = data['decoder_outputs'].float().to(device)
                    # else:
                    #     encoder_inputs = torch.cat([data['encoder_inputs'].float().to(device), encoder_inputs], dim=0)
                    #     decoder_inputs = torch.cat([data['decoder_inputs'].float().to(device), decoder_inputs], dim=0)
                    #     decoder_outputs = torch.cat([data['decoder_outputs'].float().to(device), decoder_outputs], dim=0)
                    prediction = net(encoder_inputs, decoder_inputs, train=False)
                    loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
                    run_loss += loss.item()
                test_loss += run_loss/(i+1)
                prog_valid.update(it+1, [("Training Loss", run_loss/(i+1))])
        test_loss /= config.validation_size

        if test_loss < best_val_loss:
            medel_name = checkpoint_dir + "Epoch_" + str(epoch+1) + " Loss_" + str(round(test_loss, 2))
            best_val_loss = test_loss
            torch.save(net.state_dict(), medel_name)


def prediction(config, checkpoint_filename):

    # Start testing model!

    # generate data loader
    choose = DatasetChooser(config)
    if config.datatype is not 'Human':
        prediction_dataset, bone_length = choose(prediction=True)
        prediction_loader = DataLoader(prediction_dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used'.format(device))
    torch.cuda.manual_seed(971103)
    net = ST_HMR(config, prediction=True, nbones=bone_length.shape[0]-1)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.st_lstm.parameters())))

    #if torch.cuda.device_count() > 1:
     #   print("Let's use {} GPUs!".format(str(torch.cuda.device_count())))
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(checkpoint_filename, map_location='cuda:0'))
    with torch.no_grad():
        for i, data in enumerate(prediction_loader, 0):
            x_text = data['x_text'].float().to(device)
            dec_in_test = data['dec_in_test'].float().to(device)
            y_text = data['y_text'].float().to(device).cpu().numpy()
            pred = net(x_text, dec_in_test, train=False).cpu().numpy()
            mean_error, _ = utils.mean_euler_error(config, 'default', pred, y_text[:, :10, :])


if __name__ == '__main__':

    config = config.TrainConfig('Fish', 'lie', 'all')
    train(config)