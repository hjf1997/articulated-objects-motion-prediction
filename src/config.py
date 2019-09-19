# implemented by JunfengHu
# part of the code is brown from HRM
# create time 7/20/2019

import numpy as np


class TrainConfig(object):
    """Training Configurations"""
    input_window_size = 50  # Input window size during training
    output_window_size = 10  # Output window size during training
    hidden_size = 18  # Number of hidden units for HMR
    batch_size = 32  # Batch size for training
    learning_rate = 0.001  # Learning rate
    max_epoch = 500  # Maximum training epochs
    training_size = 200  # Training iterations per epoch
    validation_size = 20  # Validation iterations per epoch
    restore = False  # Restore the trained weights or restart training from scratch
    longterm = False  # Whether we are doing super longterm prediction
    keep_prob = 0.95  # Keep probability for RNN cell weights
    context_window = 1  # Context window size in HMR, this para only applies to HMR
    encoder_recurrent_steps = 10  # Number of recurrent steps in HMR/ST_HMR
    decoder_recurrent_steps = 2  # Number of recurrent steps in ST-HMR decoder expect kinematics LSTM
    visualize = False                # visualize the predicted motion during testing
    train_model = False         # train or predict
    device_ids = [0]   #index of GPU used to train the model

    models_name = ['ERD', 'LSTM3lr', 'GRU', 'HMR', 'ST_HMR']
    model = models_name[4]

    loss_name = ['l2', 'lie', 'kinematicslie']
    loss = loss_name[1]
    """Only suitable for ST_HMR"""
    share_encoder_weights = True  # share encoder weight at each recurrent step, this param only applies to ST_HMR
    bone_dim = 3  # dimension of one bone representation, static in all datasets
    decoder_name = ['lstm', 'st_lstm', 'Kinematics_lstm']
    decoder = decoder_name[2]

    def __init__(self, dataset, datatype, action):
        self.dataset = dataset
        self.datatype = datatype
        self.filename = action
        # number of bones
        if dataset == 'Mouse':
            self.nbones = 4
        elif dataset == 'Fish':
            self.nbones = 20
        elif dataset == 'Human':
            self.nbones = 18
        elif dataset == 'CSL':
            self.nbones = 16#24

        """Define kinematic chain configurations based on dataset class. Don't modify the code below"""
        if self.dataset == 'Fish':
            self.filename = 'default'
            self.chain_config = [np.arange(0, 21)]
        elif self.dataset == 'Mouse':
            self.filename = 'default'
            self.chain_config = [np.arange(0, 5)]
        elif self.dataset == 'Human':
            self.chain_config = [np.array([0, 1, 2, 3, 4, 5]),  # leg
                                 np.array([0, 6, 7, 8, 9, 10]),  # leg
                                 np.array([0, 12, 13, 14, 15]),  # spine
                                 np.array([13, 17, 18, 19, 22, 19, 21]),  # arm
                                 np.array([13, 25, 26, 27, 30, 27, 29])]  # arm
            self.chain_loss_config = [np.array([1, 2, 3, 4, 5]),  # leg
                                 np.array([6, 7, 8, 9, 10]),  # leg
                                 np.array([0, 11, 12, 13, 14, 15]),  # spine
                                 np.array([16, 17, 18, 19, 20, 21, 22, 23]),  # arm
                                 np.array([24, 25, 26, 27, 28, 19, 30, 31])]  # arm
            self.training_chain_length = [8, 8, 18, 10, 10]
            self.index = [[6, 7, 8, 9, 10, 11, 12, 13],
                          [14, 15, 16, 17, 18, 19, 20, 21],
                          [0, 1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                          [34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
                          [44, 45, 46, 47, 48, 49, 50, 51, 52, 53]]

        elif self.dataset == 'CSL':
            self.chain_config = [np.array([0, 1, 2, 3, 4]),  # leg
                                 np.array([5, 6, 7, 8, 9]),  # leg
                                 np.array([10, 11, 12, 13, 14]),  # spine
                                 np.array([15, 16, 17, 18, 19, 20, 21]),  # arm
                                 np.array([22, 23, 24, 25, 26, 27, 28])]  # arm
            self.chain_loss_config = [np.array([0, 1, 2, 3, 4 ]),  # spine
                                 np.array([5, 6, 7, 8, 9, 10]),  # arm
                                 np.array([11, 12, 13, 14, 15])]  # arm
            self.training_chain_length = [12, 18, 18]
            self.index = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                          [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                          [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]

