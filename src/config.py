# implemented by JunfengHu
# part of the code is brown from HRM
# create time 7/20/2019

import numpy as np
import torch


class TrainConfig(object):

    """Training Configurations"""
    max_grad_norm = 5                           # Gradient clipping threshold
    input_window_size = 50                      # Input window size during training
    output_window_size = 10                     # Output window size during training
    #test_output_window = 10                    # Output window size during testing. test_output_window is overwritten by test set size when longterm is true
    hidden_size = 32                       # Number of hidden units for HMR
    decoder_hidden_size = 64            # Number of hidden units for decoders
    batch_size = 8                              # Batch size for training
    learning_rate = 0.0008                       # Learning rate
    max_epoch = 500                              # Maximum training epochs
    training_size = 50                       # Training iterations per epoch
    validation_size = 20                       # Validation iterations per epoch
    share_encoder_weights = True      # share encoder weight at each recurrent step
    restore = False                             # Restore the trained weights or restart training from scratch
    longterm = False                            # Whether we are doing super longterm prediction
    early_stop = 10                             # Stop training if validation loss do not improve after these epochs
    keep_prob = 0.8                               # Keep probability for RNN cell weights
    context_window = 1                          # Context window size in HMR
    encoder_recurrent_steps = 6                       # Number of recurrent steps in HMR
    decoder_recurrent_steps = 2            # Number of recurrent steps in LS-STLM decoder
    bone_dim = 3                                  # dimension of one bone representation
    visualize = True
    trust_gate =False
    noise_gate = False

    decoder_name = ['lstm', 'st_lstm']
    decoder = decoder_name[0]

    def __init__(self, dataset, datatype, action):
        self.dataset = dataset
        self.datatype = datatype
        self.filename = action

        if not self.trust_gate:
            self.noise_gate = False

        """Define kinematic chain configurations based on dataset class"""
        if self.dataset == 'Fish':
            self.filename = 'default'
            self.chain_config = [np.arange(0, 21)]
        elif self.dataset == 'Mouse':
            self.filename = 'default'
            self.chain_config = [np.arange(0, 5)]
        elif self.dataset == 'Human':
            self.chain_config = [np.array([0, 1, 2, 3, 4, 5]),
                                 np.array([0, 6, 7, 8, 9, 10]),
                                 np.array([0, 12, 13, 14, 15]),
                                 np.array([13, 17, 18, 19, 22, 19, 21]),
                                 np.array([13, 25, 26, 27, 30, 27, 29])]
