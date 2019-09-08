# implemented by JunfengHu
# create time: 7/20/2019

import torch
import torch.nn as nn
import torch.nn.functional as F


class ST_HMR(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_cell = torch.nn.ModuleList()
        # init encoder
        if config.share_encoder_weights is False:
            for i in range(config.encoder_recurrent_steps):
                self.encoder_cell.append(EncoderCell(config))
        else:
            shared_encoder = EncoderCell(config)
            for i in range(config.encoder_recurrent_steps):
                self.encoder_cell.append(shared_encoder)

        # init decoder
        if config.decoder == 'st_lstm':
            print('Use ST_LSTM as decoder.')
            self.decoder = ST_LSTM(config)
        elif config.decoder == 'lstm':
            print('Use LSTM as decoder.')
            self.decoder = LSTM_decoder(config)
        elif config.decoder == 'Kinematics_lstm':
            print('Use Kinematics_LSTM as decoder.')
            self.decoder = Kinematics_LSTM_decoder(config)

        self.weights_in = torch.nn.Parameter(torch.empty(config.input_size,
                                      int(config.input_size/config.bone_dim*config.hidden_size)).uniform_(-0.04, 0.04))
        self.bias_in = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size)).uniform_(-0.04, 0.04))

    def forward(self, encoder_inputs, decoder_inputs, train):
        """
        The decoder and encoder wrapper.
        :param encoder_inputs:
        :param decoder_inputs:
        :param train: train or test the model
        :return: the prediction of human motion
        """

        # [batch, config.input_window_size-1, input_size/bone_dim*hidden_size]
        h = torch.matmul(encoder_inputs, self.weights_in) + self.bias_in
        # [batch, config.input_window_size-1, hidden_size]
        h = h.view([h.shape[0], h.shape[1], int(h.shape[2]/self.config.hidden_size), self.config.hidden_size])
        # [batch, nbones, frames, hidden_state]
        h = F.dropout(h, self.config.keep_prob, train)
        c_h = torch.empty_like(h)
        c_h.copy_(h)
        c_h = F.dropout(c_h, self.config.keep_prob, train)

        p = torch.empty_like(h)
        p.copy_(h)

        # init global states
        # [batch,  nbones, hidden_size]
        g_t = torch.mean(h, 2, keepdim=True).expand_as(h)
        c_g_t = torch.mean(c_h, 2, keepdim=True).expand_as(c_h)

        # [batch, input_window_size-1, hidden_size]
        g_s = torch.mean(h, 1, keepdim=True).expand_as(h)
        c_g_s = torch.mean(c_h, 1, keepdim=True).expand_as(c_h)

        for rec in range(self.config.encoder_recurrent_steps):
            hidden_states, cell_states, global_t_state, g_t, c_g_t, g_s, c_g_s = self.encoder_cell[rec](h, c_h, p, g_t, c_g_t, g_s, c_g_s, train)
        prediction = self.decoder(hidden_states, cell_states, global_t_state, decoder_inputs)
        return prediction


class EncoderCell(nn.Module):
    """
    ST_HMR encoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        """h update gates"""
        # input forget gate
        self.Ui = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wti = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsi = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zti = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsi = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bi = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # left time forget gate
        self.Ult = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wtlt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wslt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ztlt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.blt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forward time forget gate
        self.Uft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wtft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ztft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # right time forget gate
        self.Urt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wtrt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsrt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ztrt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.brt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # space forget gate
        self.Us = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wts = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zts = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global time forgate gate
        self.Ugt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wtgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ztgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global space fotgate gate
        self.Ugs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wtgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ztgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Uo = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bo = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # c_hat gate
        self.Uc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wtc = torch.nn.Parameter( torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ztc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_t update gates"""
        # forget gates for h
        self.Wgtf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgtf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgtf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g
        self.Wgtg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgtg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgtg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_s update gates"""
        # forget gates for h
        self.Wgsf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g
        self.Wgsg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

    def forward(self, h, c_h, p, g_t, c_g_t, g_s, c_g_s, train):
        """
        :param h: hidden states of [batch, input_window_size-1, nbones, hidden_size]
        :param c_h: cell states of  [batch, input_window_size-1, nbones, hidden_size]
        :param p: pose of  [batch, input_window_size-1, nbones, hidden_size]
        :param g_t:
        :param c_g_t:
        :param g_s:
        :param c_g_s:
        :param train:
        :return:
        """

        padding_t = torch.zeros_like(h[:, 0:1, :, :])
        padding_s = torch.zeros_like(h[:, :, 0:1, :])

        """Update h"""

        h_t_before = torch.cat((padding_t, h[:, :-1, :, :]), dim=1)
        h_t_after = torch.cat((h[:, 1:, :, :], padding_t), dim=1)
        # [batch, input_window_size-1, nbones, hidden_size*3]
        h_t_before_after = torch.cat((h_t_before, h, h_t_after), dim=3)

        c_t_before = torch.cat((padding_t, c_h[:, :-1, :, :]), dim=1)
        c_t_after = torch.cat((c_h[:, 1:, :, :], padding_t), dim=1)
        # [batch, input_window_size-1, nbones, hidden_size*3]

        h_s_before = torch.cat((padding_s, h[:, :, :-1, :]), dim=2)
        c_s_before = torch.cat((padding_s, c_h[:, :, :-1, :]), dim=2)

        # forget gates for h
        i_n = torch.sigmoid(torch.matmul(p, self.Ui) + torch.matmul(h_t_before_after, self.Wti)
                            + torch.matmul(h_s_before, self.Wsi) + torch.matmul(g_t, self.Zti)
                            + torch.matmul(g_s, self.Zsi) + self.bi)
        f_lt_n = torch.sigmoid(torch.matmul(p, self.Ult) + torch.matmul(h_t_before_after, self.Wtlt)
                               + torch.matmul(h_s_before, self.Wslt) + torch.matmul(g_t, self.Ztlt)
                               + torch.matmul(g_s, self.Zslt) + self.blt)
        f_ft_n = torch.sigmoid(torch.matmul(p, self.Uft) + torch.matmul(h_t_before_after, self.Wtft)
                               + torch.matmul(h_s_before, self.Wsft) + torch.matmul(g_t, self.Ztft)
                               + torch.matmul(g_s, self.Zslt) + self.bft)
        f_rt_n = torch.sigmoid(torch.matmul(p, self.Urt) + torch.matmul(h_t_before_after, self.Wtrt)
                               + torch.matmul(h_s_before, self.Wsrt) + torch.matmul(g_t, self.Ztrt)
                               + torch.matmul(g_s, self.Zsrt) + self.brt)
        f_s_n = torch.sigmoid(torch.matmul(p, self.Us) + torch.matmul(h_t_before_after, self.Wts)
                              + torch.matmul(h_s_before, self.Wss) + torch.matmul(g_t, self.Zts)
                              + torch.matmul(g_s, self.Zss) + self.bs)
        f_gt_n = torch.sigmoid(torch.matmul(p, self.Ugt) + torch.matmul(h_t_before_after, self.Wtgt)
                               + torch.matmul(h_s_before, self.Wsgt) + torch.matmul(g_t, self.Ztgt)
                               + torch.matmul(g_s, self.Zsgt) + self.bgt)
        f_gs_n = torch.sigmoid(torch.matmul(p, self.Ugs) + torch.matmul(h_t_before_after, self.Wtgs)
                               + torch.matmul(h_s_before, self.Wsgs) + torch.matmul(g_t, self.Ztgs)
                               + torch.matmul(g_s, self.Zsgs) + self.bgs)
        o_n = torch.sigmoid(torch.matmul(p, self.Uo) + torch.matmul(h_t_before_after, self.Wto)
                            + torch.matmul(h_s_before, self.Wso) + torch.matmul(g_t, self.Zto)
                            + torch.matmul(g_s, self.Zso) + self.bo)
        c_n = torch.tanh(torch.matmul(p, self.Uc) + torch.matmul(h_t_before_after, self.Wtc)
                         + torch.matmul(h_s_before, self.Wsc) + torch.matmul(g_t, self.Ztc)
                         + torch.matmul(g_s, self.Zsc) + self.bc)

        c_h = (f_lt_n * c_t_before) + (f_ft_n * c_h) + (f_rt_n * c_t_after) + (f_s_n * c_s_before)\
                        + (f_gt_n * c_g_t) + (f_gs_n * c_g_s) + (c_n * i_n)

        h = o_n * torch.tanh(c_h)

        c_h = torch.dropout(c_h, self.config.keep_prob, train)
        h = torch.dropout(h, self.config.keep_prob, train)
        """Update g_t"""
        g_t_hat = torch.mean(h, 1, keepdim=True).expand_as(h)
        f_gtf_n = torch.sigmoid(torch.matmul(g_t, self.Wgtf) + torch.matmul(g_t_hat, self.Zgtf) + self.bgtf)
        f_gtg_n = torch.sigmoid(torch.matmul(g_t, self.Wgtg) + torch.matmul(g_t_hat, self.Zgtg) + self.bgtg)
        o_gt_n = torch.sigmoid(torch.matmul(g_t, self.Wgto) + torch.matmul(g_t_hat, self.Zgto) + self.bgto)

        c_g_t = torch.sum(f_gtf_n * c_h, dim=1, keepdim=True).expand_as(c_h) + c_g_t * f_gtg_n
        g_t = o_gt_n * torch.tanh(c_g_t)

        """Update g_s"""
        g_s_hat = torch.mean(h, 2, keepdim=True).expand_as(h)
        f_gsf_n = torch.sigmoid(torch.matmul(g_s, self.Wgsf) + torch.matmul(g_s_hat, self.Zgsf) + self.bgsf)
        f_gsg_n = torch.sigmoid(torch.matmul(g_s, self.Wgsg) + torch.matmul(g_s_hat, self.Zgsg) + self.bgsg)
        o_gs_n = torch.sigmoid(torch.matmul(g_s, self.Wgso) + torch.matmul(g_s_hat, self.Zgso) + self.bgso)

        c_g_s = torch.sum(f_gsf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_s * f_gsg_n
        g_s = o_gs_n * torch.tanh(c_g_s)
        hidden_states = h.view([h.shape[0], h.shape[1], -1])
        cell_states = c_h.view([c_h.shape[0], c_h.shape[1], -1])
        global_t_state = g_t[:, 1, :, :].view([g_t.shape[0], -1])

        return hidden_states, cell_states, global_t_state, g_t, c_g_t, g_s, c_g_s


class Kinematics_LSTM_decoder(nn.Module):

    def __init__(self, config):
        """
        This decoder only apply to h3.6m dataset.
        :param config:
        """
        super().__init__()
        self.config = config
        self.seq_length_out = config.output_window_size
        self.nbones = config.nbones
        self.lstm = nn.ModuleList()
        self.weights_out_spine = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.training_chain_length[2]).uniform_(-0.04, 0.04))
        self.bias_out_spine = torch.nn.Parameter(torch.empty(config.training_chain_length[2]).uniform_(-0.04, 0.04))
        self.weights_out_leg1 = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.training_chain_length[0]).uniform_(-0.04, 0.04))
        self.bias_out_leg1 = torch.nn.Parameter(torch.empty(config.training_chain_length[0]).uniform_(-0.04, 0.04))
        self.weights_out_leg2 = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.training_chain_length[1]).uniform_(-0.04, 0.04))
        self.bias_out_leg2 = torch.nn.Parameter(torch.empty(config.training_chain_length[1]).uniform_(-0.04, 0.04))
        self.weights_out_arm1 = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.training_chain_length[3]).uniform_(-0.04, 0.04))
        self.bias_out_arm1 = torch.nn.Parameter(torch.empty(config.training_chain_length[3]).uniform_(-0.04, 0.04))
        self.weights_out_arm2 = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.training_chain_length[4]).uniform_(-0.04, 0.04))
        self.bias_out_arm2 = torch.nn.Parameter(torch.empty(config.training_chain_length[4]).uniform_(-0.04, 0.04))
        # LSTM First layer
        self.lstm.append(nn.LSTMCell(config.input_size, int(config.input_size / config.bone_dim * config.hidden_size)))
        # Kinematics LSTM layer
        spine = nn.LSTMCell(int(config.input_size / config.bone_dim * config.hidden_size), int(config.input_size / config.bone_dim * config.hidden_size))
        self.lstm.append(spine)
        arm = nn.LSTMCell(int(config.input_size / config.bone_dim * config.hidden_size), int(config.input_size / config.bone_dim * config.hidden_size))
        self.lstm.append(arm)
        self.lstm.append(arm)
        leg = nn.LSTMCell(int(config.input_size / config.bone_dim * config.hidden_size), int(config.input_size / config.bone_dim * config.hidden_size))
        self.lstm.append(leg)
        self.lstm.append(leg)
        self.lstm_layer = 6

    def forward(self, hidden_states, cell_states, global_t_state, p):

        # define decoder hidden states and cell states
        h = []
        c_h = []
        pre = torch.zeros([hidden_states.shape[0], self.seq_length_out, self.config.input_size], device=p.device)
        for i in range(self.lstm_layer):
            h.append(torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones * self.config.hidden_size,
                              device=p.device))
            c_h.append(torch.zeros_like(h[i]))
            # feed init hidden states and cell states into h and c_h
            if i == 0:
                h_t = hidden_states
            elif i == 1:
                h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1)
            if i < 2:
                h[i][:, 0, :] = h_t.mean(dim=1)
                c_h[i][:, 0, :] = torch.mean(cell_states, dim=1)

        for frame in range(self.seq_length_out):
            for i in range(self.lstm_layer):
                cell = self.lstm[i]
                if i == 0:
                    if frame == 0:
                        input = p[:, 0, :]
                        input_first = p[:, 0, :]
                    else:
                        input = pre[:, frame - 1, :].clone()
                        input_first = pre[:, frame - 1, :].clone()
                else:
                    if i == (3 or 4 or 5):
                        input = h[1][:, frame + 1, :].clone()
                    else:
                        input = h[i-1][:, frame + 1, :].clone()
                h[i][:, frame + 1, :], c_h[i][:, frame + 1, :] \
                    = cell(input, (h[i][:, frame, :].clone(), c_h[i][:, frame, :].clone()))

                if i == 1:
                    pre[:, frame, self.config.index[2]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_spine) + \
                                   self.bias_out_spine + input_first[:, self.config.index[2]]
                elif i == 2:
                    pre[:, frame, self.config.index[0]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_leg1) + \
                                   self.bias_out_leg1 + input_first[:, self.config.index[0]]
                elif i == 3:
                    pre[:, frame, self.config.index[1]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_leg2) + \
                                   self.bias_out_leg2 + input_first[:, self.config.index[1]]
                elif i == 4:
                    pre[:, frame, self.config.index[3]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_arm1) + \
                                   self.bias_out_arm1 + input_first[:, self.config.index[3]]
                elif i == 5:
                    pre[:, frame, self.config.index[4]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_arm2) + \
                                   self.bias_out_arm2 + input_first[:, self.config.index[4]]

        return pre


class LSTM_decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_length_out = config.output_window_size
        self.nbones = config.nbones
        self.lstm = nn.ModuleList()
        self.weights_out = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.input_size).uniform_(-0.04, 0.04))
        self.bias_out = torch.nn.Parameter(torch.empty(config.input_size).uniform_(-0.04, 0.04))
        for i in range(config.decoder_recurrent_steps):
            if i == 0:
                self.lstm.append(nn.LSTMCell(config.input_size, int(config.input_size/config.bone_dim*config.hidden_size)))
            else:
                self.lstm.append(nn.LSTMCell(int(config.input_size/config.bone_dim*config.hidden_size), int(config.input_size/config.bone_dim*config.hidden_size)))

    def forward(self, hidden_states, cell_states, global_t_state, p):

        # define decoder hidden states and cell states
        h = []
        c_h = []
        pre = torch.zeros([hidden_states.shape[0], self.seq_length_out, self.config.input_size], device=p.device)
        for i in range(self.config.decoder_recurrent_steps):
            h.append(torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones * self.config.hidden_size,
                              device=p.device))
            c_h.append(torch.zeros_like(h[i]))
            # feed init hidden states and cell states into h and c_h
            if i == 0:
                h_t = hidden_states
            elif i == 1:
                h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1)
            else:
                raise Exception('The max decoder_recurrent_steps is 2!')
            h[i][:, 0, :] = h_t.mean(dim=1)
            c_h[i][:, 0, :] = torch.mean(cell_states, dim=1)

        for frame in range(self.seq_length_out):
            for i in range(self.config.decoder_recurrent_steps):
                cell = self.lstm[i]
                if i == 0:
                    if frame == 0:
                        input = p[:, 0, :]
                        input_first = p[:, 0, :]
                    else:
                        input = pre[:, frame - 1, :].clone()
                        input_first = pre[:, frame - 1, :].clone()
                else:
                    input = h[i - 1][:, frame + 1, :].clone()
                h[i][:, frame + 1, :], c_h[i][:, frame + 1, :] \
                    = cell(input, (h[i][:, frame, :].clone(), c_h[i][:, frame, :].clone()))
            pre[:, frame, :] = torch.matmul(h[-1][:, frame + 1, :].clone(), self.weights_out) + \
                           self.bias_out + input_first
        return pre


class ST_LSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config = config.nbones
        recurrent_cell_box = torch.nn.ModuleList()
        self.seq_length_out = config.output_window_size
        self.weights_out = torch.nn.Parameter(torch.empty(config.hidden_size, config.bone_dim).uniform_(-0.04, 0.04))
        self.bias_out = torch.nn.Parameter(torch.empty(config.bone_dim).uniform_(-0.04, 0.04))

        for i in range(config.decoder_recurrent_steps):
            cells = torch.nn.ModuleList()
            for bone in range(config.nbones):
                if i == 0:
                    cell = ST_LSTMCell(config.bone_dim, config.hidden_size)
                else:
                    cell = ST_LSTMCell(config.hidden_size, config.hidden_size)
                cells.append(cell)
            recurrent_cell_box.append(cells)
        self.recurrent_cell_box = recurrent_cell_box

    def forward(self, hidden_states, cell_states, global_t_state, global_s_state, p):
        """
        :param hidden_states:  [batch, input_window_size-1, nbones, hidden_size]
        :param cell_states: [batch, input_window_size-1, nbones, hidden_size]
        :param global_t_state: [batch,  nbones, hidden_size]
        :param global_s_state: [batch, input_window_size-1, hidden_size]
        :param p: [batch, input_window_size-1, nbones, hidden_size]
        :return:
        """

        # define decoder hidden states and cell states
        h = []
        c_h = []
        pre = torch.zeros([hidden_states.shape[0], self.seq_length_out, self.nbones, self.config.bone_dim], device=p.device)
        p = p.view([p.shape[0], p.shape[1], self.nbones, self.config.bone_dim])
        for i in range(self.config.decoder_recurrent_steps):
            h.append(torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones + 1, self.config.hidden_size,
                              device=p.device))
            c_h.append(torch.zeros_like(h[i]))
            # feed init hidden states and cell states into h and c_h
            if i == 0:
                h_t = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.nbones, self.config.hidden_size)
            elif i == 1:
                h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1).view(hidden_states.shape[0], hidden_states.shape[1]+1, self.nbones, self.config.hidden_size)
            else:
                print('The max decoder num is 2!')

            h[i][:, 0, 1:, :] = torch.mean(h_t, dim=1)
            c_h[i][:, 0, 1:, :] = torch.mean(cell_states.view(cell_states.shape[0], cell_states.shape[1], self.nbones, self.config.hidden_size), dim=1)

        for frame in range(self.seq_length_out):
            for i in range(self.config.decoder_recurrent_steps):
                for bone in range(self.nbones):
                    cell = self.recurrent_cell_box[i][bone]
                    if i == 0:
                        if frame == 0:
                            input = p[:, 0, bone, :]
                            input_first = p[:, 0, bone, :]
                        else:
                            input = pre[:, frame - 1, bone, :].clone()
                            input_first = pre[:, frame - 1, bone, :].clone()
                    else:
                        input = h[i - 1][:, frame + 1, bone, :].clone()

                    h[i][:, frame+1, bone+1, :], c_h[i][:, frame+1, bone+1, :] \
                        = cell(input, h[i][:, frame, bone+1, :].clone(),
                            h[i][:, frame+1, bone, :].clone(), c_h[i][:, frame, bone+1, :].clone(), c_h[i][:, frame+1, bone, :].clone())
            pre[:, frame, :, :] = torch.matmul(h[-1][:, frame + 1, :, :].clone(), self.weights_out) + self.bias_out + input_first
        pre_c = c_h[-1][:, 1:, 1:, :].view([c_h[-1][:, 1:, 1:, :].shape[0], c_h[-1][:, 1:, 1:, :].shape[1], -1])
        pre = pre.view([pre.shape[0], pre.shape[1], -1])

        return pre, pre_c


class ST_LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        # input gate
        self.Ui = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wti = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wsi = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bi = torch.nn.Parameter(torch.randn(hidden_size))
        # space forget gate
        self.Us = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wts = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wss = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bs = torch.nn.Parameter(torch.randn(hidden_size))
        # time forget gate
        self.Ut = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wtt = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wst = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bt = torch.nn.Parameter(torch.randn(hidden_size))
        # output gate
        self.Uo = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wto = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wso = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bo = torch.nn.Parameter(torch.randn(hidden_size))
        # c_hat gate
        self.Uc = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wtc = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wsc = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bc = torch.nn.Parameter(torch.randn(hidden_size))

    def forward(self, x, h_t, h_s, c_t, c_s):

        i_n = torch.sigmoid(torch.matmul(x, self.Ui) + torch.matmul(h_t, self.Wti) + torch.matmul(h_s, self.Wsi) + self.bi)
        f_s_n = torch.sigmoid(torch.matmul(x, self.Us) + torch.matmul(h_t, self.Wts) + torch.matmul(h_s, self.Wss) + self.bs)
        f_t_n = torch.sigmoid(torch.matmul(x, self.Ut) + torch.matmul(h_t, self.Wtt) + torch.matmul(h_s, self.Wst) + self.bt)
        o_n = torch.sigmoid(torch.matmul(x, self.Uo) + torch.matmul(h_t, self.Wto) + torch.matmul(h_s, self.Wso) + self.bo)
        c_n = torch.tanh(torch.matmul(x, self.Uc) + torch.matmul(h_t, self.Wtc) + torch.matmul(h_s, self.Wsc) + self.bc)

        c_h = (i_n * c_n) + (f_t_n * c_t) + (f_s_n * c_s)
        h = o_n * torch.tanh(c_h)

        return h, c_h
