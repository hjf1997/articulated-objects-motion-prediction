# implemented by JunfengHu
# created time: 7/20/2019

import torch
from torch.utils.data import DataLoader
import numpy as np
from utils import Progbar
from choose_dataset import DatasetChooser
from loss import linearizedlie_loss, l2_loss
import utils
import scipy.io as sio
import torch.optim as optim
import re
import config
from plot_animation import plot_animation
import os
from STLN import ST_HMR
from HMR import HMR


def train(config):

    print('Start Training the Model!')

    # generate data loader
    choose = DatasetChooser(config)
    train_dataset, bone_length = choose(train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset, _ = choose(train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    prediction_dataset, bone_length = choose(prediction=True)
    x_test, y_test, dec_in_test = prediction_dataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used to save parameters'.format(device))
    torch.cuda.manual_seed(112858)
    net = ST_HMR(config)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.decoder.parameters())))
    # to = 0.0
    # for p in net.decoder.parameters():
    #     to += p.abs().sum()
    # print(to)
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(str(torch.cuda.device_count())))
    net = torch.nn.DataParallel(net)# device_ids=[1, 2, 3]) # device_ids=[1, 2, 3]
    #net.load_state_dict(torch.load('./model/Epoch_101 Loss_0.0505.pth'))
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

    best_error = float('inf')
    #best_error = np.array([float('inf'), float('inf'), float('inf'), float('inf')])
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
                #loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
                loss = l2_loss(prediction, decoder_outputs)
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
                        loss = l2_loss(prediction, decoder_outputs)
                        #loss = linearizedlie_loss(prediction, decoder_outputs, bone_length)
                        run_loss += loss.item()
                    prog_valid.update(it+1, [("Training Loss", run_loss/(i+1))])

        # Test prediction
        actions = list(x_test.keys())
        y_predict = {}
        with torch.no_grad():
            for act in actions:
                x_test_ = torch.from_numpy(x_test[act]).float().to(device)
                dec_in_test_ = torch.from_numpy(dec_in_test[act]).float().to(device)
                pred = net(x_test_, dec_in_test_, train=False)
                pred = pred.cpu().numpy()
                y_predict[act] = pred

        error_actions = 0.0
        for act in actions:
            if config.datatype == 'lie':
                mean_error, _ = utils.mean_euler_error(config, act, y_predict[act], y_test[act])
                error = mean_error[[1, 3, 7, 9]]
            error_actions += error.mean()
        error_actions /= len(actions)
        if error_actions < best_error:
            best_error = error_actions
            torch.save(net.state_dict(),'./model/Epoch_' + str(epoch + 1) + ' Error' + str(round(best_error, 4)) + '.pth')
        # print('Current best mean_error: '+str(round(best_error[0], 2)) + ' & ' + str(round(best_error[1], 2)) + ' & ' + str(round(best_error[2], 2))
        #       + ' & ' + str(round(best_error[3], 2)))


def prediction(config, checkpoint_filename):

    # Start testing model!

    # generate data loader
    config.output_window_size = 75
    choose = DatasetChooser(config)
    if config.dataset is 'Human':
        _, _ = choose(train=True) # get mean value etc for unnorm

    prediction_dataset, bone_length = choose(prediction=True)
    x_test, y_test, dec_in_test = prediction_dataset

    actions = list(x_test.keys())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used to save parameters'.format(device))
    torch.cuda.manual_seed(973)
    net = HMR(config)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.decoder.parameters())))

    #if torch.cuda.device_count() > 1:
    #    print("Let's use {} GPUs!".format(str(torch.cuda.device_count())))
    net = torch.nn.DataParallel(net)
    #net.load_state_dict(torch.load(checkpoint_filename, map_location='cuda:0'))
    y_predict = {}
    with torch.no_grad():
        # This loop runs only once.
        for act in actions:
            x_test_ = torch.from_numpy(x_test[act]).float().to(device)
            dec_in_test_ = torch.from_numpy(dec_in_test[act]).float().to(device)
            pred = net(x_test_, dec_in_test_, train=False)
            pred = pred.cpu().numpy()
            y_predict[act] = pred

    for act in actions:
        if config.datatype == 'lie':
            mean_error, _ = utils.mean_euler_error(config, act, y_predict[act], y_test[act])

            for i in range(y_predict[act].shape[0]):
                if config.dataset == 'Human':
                    y_p = utils.unNormalizeData(y_predict[act][i], config.data_mean, config.data_std, config.dim_to_ignore)
                    y_t = utils.unNormalizeData(y_test[act][i], config.data_mean, config.data_std, config.dim_to_ignore)
                    expmap_all = utils.revert_coordinate_space(np.vstack((y_t, y_p)), np.eye(3), np.zeros(3))
                    y_p = expmap_all[config.output_window_size:]
                    y_t = expmap_all[:config.output_window_size]
                else:
                    y_p = y_predict[act][i]
                    y_t = y_test[act][i]

                y_p_xyz = utils.fk(y_p, config, bone_length)
                y_t_xyz = utils.fk(y_t, config, bone_length)

                filename = act + '_' + str(i)
                if config.visualize:
                    # 蓝色是我的
                    pre_plot = plot_animation(y_t_xyz, y_p_xyz, config, filename)
                    pre_plot.plot()

if __name__ == '__main__':

    config = config.TrainConfig('Human', 'lie', 'all')
    #prediction(config, './model/Epoch_13 Error1.9763.pth')
    train(config)