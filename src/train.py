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
from plot_animation import plot_animation
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
    prediction_dataset, bone_length = choose(prediction=True)
    prediction_loader = DataLoader(prediction_dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used to save parameters'.format(device))
    torch.cuda.manual_seed(971103)
    net = ST_HMR(config, bone_length.shape[0]-1)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.decoder.parameters())))

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(str(torch.cuda.device_count())))
    net = torch.nn.DataParallel(net) # device_ids=[1, 2, 3]
   # net.load_state_dict(torch.load('./model/Epoch_101 Loss_0.0505.pth'))
    # save_model = torch.load(r'./model/_Epoch_242 Loss_0.0066.pth')
    # model_dict = net.state_dict()
    # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # net.load_state_dict(model_dict)

    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    # Save model
    checkpoint_dir = './model/'
    if not (os.path.exists(checkpoint_dir)):
        os.makedirs(checkpoint_dir)

    best_val_loss = float('inf')
    best_error = np.array([float('inf'), float('inf'), float('inf'), float('inf')])
    for epoch in range(config.max_epoch):
        print("At epoch:{}".format(str(epoch + 1)))
        prog = Progbar(target=config.training_size)
        prog_valid = Progbar(target=config.validation_size)

        # Train
        #with torch.autograd.set_detect_anomaly(True):
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
                prediction = net(encoder_inputs, decoder_inputs, train=True)
                #print('___')
                #print(torch.sum(prediction))
                loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
                run_loss += loss.item()
                net.zero_grad()
                loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
            prog.update(it + 1, [("Training Loss", run_loss/(i+1))])

        # Test
        with torch.no_grad():
            for it in range(config.validation_size):
                    run_loss = 0.0
                    for i, data in enumerate(test_loader, 0):
                        encoder_inputs = data['encoder_inputs'].float().to(device)
                        decoder_inputs = data['decoder_inputs'].float().to(device)
                        decoder_outputs = data['decoder_outputs'][:, :10, :].float().to(device)
                        prediction = net(encoder_inputs, decoder_inputs, train=False)
                        loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
                        run_loss += loss.item()
                    prog_valid.update(it+1, [("Training Loss", run_loss/(i+1))])

        # Test prediction
        av_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(prediction_loader, 0):
                x_test = data['x_test'].float().to(device)
                dec_in_test = data['dec_in_test'].float().to(device)
                y_test = data['y_test'].float().to(device)  # .cpu().numpy()
                pred = net(x_test, dec_in_test, train=False)  # .cpu().numpy()
                loss = linearizedlie_loss(pred, y_test[:, :10, :], bone_length)
                av_loss += loss.item()
                y_test = y_test.cpu().numpy()
                pred = pred.cpu().numpy()
                mean_error, _ = utils.mean_euler_error(config, 'default', pred, y_test[:, :10, :])
                error = mean_error[[1, 3, 7, 9]]

            if error.sum() < best_error.sum():
                best_error = error
                torch.save(net.state_dict(),'./model/Epoch_' + str(epoch + 1) + ' Loss_' + str(round(av_loss / (i + 1), 4)) + '.pth')
            print('Current best mean_error: '+str(round(best_error[0], 2)) + ' & ' + str(round(best_error[1], 2)) + ' & ' + str(round(best_error[2], 2))
                  + ' & ' + str(round(best_error[3], 2)))


def prediction(config, checkpoint_filename):

    # Start testing model!

    # generate data loader
    config.output_window_size = 75
    choose = DatasetChooser(config)
    if config.datatype is not 'Human':
        prediction_dataset, bone_length = choose(prediction=True)
        prediction_loader = DataLoader(prediction_dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used to save parameters'.format(device))
    torch.cuda.manual_seed(973)
    net = ST_HMR(config, nbones=bone_length.shape[0]-1)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.decoder.parameters())))

    #if torch.cuda.device_count() > 1:
    #    print("Let's use {} GPUs!".format(str(torch.cuda.device_count())))
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(checkpoint_filename, map_location='cuda:0'))
    with torch.no_grad():
        # This loop runs only once.
        for i, data in enumerate(prediction_loader, 0):
            x_test = data['x_test'].float().to(device)
            dec_in_test = data['dec_in_test'].float().to(device)
            y_test = data['y_test'].float().to(device)#.cpu().numpy()
            pred = net(x_test, dec_in_test, train=False)#.cpu().numpy()
            #loss = linearizedlie_loss(pred, y_test[:, :100, :], bone_length)
            y_test = y_test.cpu().numpy()
            pred = pred.cpu().numpy()

    if config.datatype == 'lie':
        print(pred.shape)
        print(y_test.shape)
        mean_error, _ = utils.mean_euler_error(config, 'default', pred, y_test[:, :100, :])

        for i in range(pred.shape[0]):
            if config.datatype == 'Human':
                pass
            else:
                y_p = pred[i]
                #y_p = np.concatenate([x_test[i], y_p], axis=0)
                y_t = y_test[i]
                #y_t = np.concatenate([x_test[i], y_t], axis=0)

            y_p_xyz = utils.fk(y_p, config, bone_length)
            y_t_xyz = utils.fk(y_t, config, bone_length)

            if config.visualize:
                pre_plot = plot_animation(y_t_xyz, y_p_xyz, config, None)
                pre_plot.plot()

if __name__ == '__main__':

    config = config.TrainConfig('Mouse', 'lie', 'all')
    #prediction(config, './model/mouse.pth')
    train(config)