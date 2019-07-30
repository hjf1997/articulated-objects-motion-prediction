# implemented by JunfengHu
# create time: 7/20/2019

import torch
import numpy as np
import config
import utils
import torch.nn as nn
import torch.nn.functional as F


class ST_HMR(nn.Module):

    def __init__(self, config, prediction, nbones):
        super().__init__()
        self.config = config
        self.encoder_cell = torch.nn.ModuleList()
        # init encoder
        if config.share_encoder_weights is False:
            for i in range(config.encoder_recurrent_steps):
                self.encoder_cell.append(EncoderCell(config, nbones))
        else:
            shared_encoder = EncoderCell(config, nbones)
            for i in range(config.encoder_recurrent_steps):
                self.encoder_cell.append(shared_encoder)

        # init decoder
        self.st_lstm = ST_LSTM(config, prediction, nbones)

        self.weights_in = torch.nn.Parameter(torch.empty(config.input_size,
                                      int(config.input_size/config.bone_dim*config.hidden_size)).uniform_(-0.04, 0.04))
        self.bias_in = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size)).uniform_(-0.04, 0.04))
        self.weights_out = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.input_size).uniform_(-0.04, 0.04))
        self.bias_out = torch.nn.Parameter(torch.empty(config.input_size).uniform_(-0.04, 0.04))

    def forward(self, encoder_inputs, decoder_inputs, train):

        # [batch, config.input_window_size-1, input_size/bone_dim*hidden_size]
        h = torch.matmul(encoder_inputs, self.weights_in) + self.bias_in
        # [batch, config.input_window_size-1, nbones, hidden_size]
        h = h.view([h.shape[0], h.shape[1], int(h.shape[2]/self.config.hidden_size), self.config.hidden_size])
        # [batch, nbones, frames, hidden_state]
        # This step is for the broadcasted operation in encoder and will be
        # permuted to original form when feeding into decoder
        h = h.permute([0, 2, 1, 3])
        h = torch.dropout(h, self.config.keep_prob, train)
        c_h = torch.empty_like(h)
        c_h.copy_(h)
        c_h = torch.dropout(c_h, self.config.keep_prob, train)

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
            hidden_states, cell_states, global_t_state, global_s_state, g_t, c_g_t, g_s, c_g_s = self.encoder_cell[rec](h, c_h, p, g_t, c_g_t, g_s, c_g_s, train)

        hidden_states = hidden_states.permute([0, 2, 1, 3])
        cell_states = cell_states.permute([0, 2, 1, 3])

        decoder_p = torch.matmul(decoder_inputs, self.weights_in) + self.bias_in
        decoder_p = decoder_p.view([decoder_p.shape[0], decoder_p.shape[1], int(decoder_p.shape[2]/self.config.hidden_size), self.config.hidden_size])
        prediction, _ = self.st_lstm(hidden_states, cell_states, global_t_state, global_s_state, decoder_p)
        prediction = prediction.view([prediction.shape[0], prediction.shape[1], prediction.shape[2]*prediction.shape[3]])
        prediction = torch.matmul(prediction, self.weights_out) + self.bias_out
        return prediction


class EncoderCell(nn.Module):

    def __init__(self, config, nbones):
        super().__init__()
        self.config = config

        """h update gates"""
        # input forget gate
        self.Ui = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wti = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wsi = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zti = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zsi = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bi = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # left time forget gate
        self.Ult = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wtlt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wslt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Ztlt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zslt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.blt = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # forward time forget gate
        self.Uft = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wtft = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wsft = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Ztft = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zsft = torch.nn.Parameter(torch.randn(nbones,  self.config.hidden_size, self.config.hidden_size))
        self.bft = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # right time forget gate
        self.Urt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wtrt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wsrt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Ztrt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zsrt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.brt = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # space forget gate
        self.Us = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wts = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wss = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zts = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zss = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bs = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # global time forgate gate
        self.Ugt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wtgt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wsgt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Ztgt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zsgt = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bgt = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # global space fotgate gate
        self.Ugs = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wtgs = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wsgs = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Ztgs = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zsgs = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bgs = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # output gate
        self.Uo = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wto = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wso = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zto = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zso = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bo = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # c_hat gate
        self.Uc = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Wtc = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size*3, self.config.hidden_size))
        self.Wsc = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Ztc = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zsc = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bc = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        """g_t update gates"""
        # forget gates for h
        self.Wgtf = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zgtf = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bgtf = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # forget gate for g
        self.Wgtg = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zgtg = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bgtg = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        #output gate
        self.Wgto = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zgto = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bgto = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        """g_s update gates"""
        # forget gates for h
        self.Wgsf = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zgsf = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bgsf = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # forget gate for g
        self.Wgsg = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zgsg = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bgsg = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

        # output gate
        self.Wgso = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.Zgso = torch.nn.Parameter(torch.randn(nbones, self.config.hidden_size, self.config.hidden_size))
        self.bgso = torch.nn.Parameter(torch.randn(nbones, 1, self.config.hidden_size))

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

        padding_t = torch.zeros_like(h[:, :, 0:1, :])
        padding_s = torch.zeros_like(h[:, 0:1, :, :])

        """Update h"""

        h_t_before = torch.cat((padding_t, h[:, :, :-1, :]), dim=2)
        h_t_after = torch.cat((h[:, :, 1:, :], padding_t), dim=2)
        # [batch, input_window_size-1, nbones, hidden_size*3]
        h_t_before_after = torch.cat((h_t_before, h, h_t_after), dim=3)

        c_t_before = torch.cat((padding_t, c_h[:, :, :-1, :]), dim=2)
        c_t_after = torch.cat((c_h[:, :, 1:, :], padding_t), dim=2)
        # [batch, input_window_size-1, nbones, hidden_size*3]
        c_t_before_after = torch.cat((c_t_before, c_h, c_t_after), dim=3)

        h_s_before = torch.cat((padding_s, h[:, :-1, :, :]), dim=1)
        c_s_before = torch.cat((padding_s, c_h[:, :-1, :, :]), dim=1)

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
        g_t_hat = torch.mean(h, 2, keepdim=True).expand_as(h)
        f_gtf_n = torch.sigmoid(torch.matmul(g_t, self.Wgtf) + torch.matmul(g_t_hat, self.Zgtf) + self.bgtf)
        f_gtg_n = torch.sigmoid(torch.matmul(g_t, self.Wgtg) + torch.matmul(g_t_hat, self.Zgtg) + self.bgtg)
        o_gt_n = torch.sigmoid(torch.matmul(g_t, self.Wgto) + torch.matmul(g_t_hat, self.Zgto) + self.bgto)

        c_g_t = torch.sum(f_gtf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_t * f_gtg_n
        g_t = o_gt_n * c_g_t

        """Update g_s"""
        g_s_hat = torch.mean(h, 1, keepdim=True).expand_as(h)
        f_gsf_n = torch.sigmoid(torch.matmul(g_s, self.Wgsf) + torch.matmul(g_s_hat, self.Zgsf) + self.bgsf)
        f_gsg_n = torch.sigmoid(torch.matmul(g_s, self.Wgsg) + torch.matmul(g_s_hat, self.Zgsg) + self.bgsg)
        o_gs_n = torch.sigmoid(torch.matmul(g_s, self.Wgso) + torch.matmul(g_s_hat, self.Zgso) + self.bgso)

        c_g_s = torch.sum(f_gsf_n * c_h, dim=1, keepdim=True).expand_as(c_h) + c_g_s * f_gsg_n
        g_s = o_gs_n * torch.tanh(c_g_s)

        return h, c_h, g_t[:, :, 1, :], g_s[:, 1, :, :], g_t, c_g_t, g_s, c_g_s


class ST_LSTM(nn.Module):

    def __init__(self, config, prediction, nbones):
        super().__init__()
        self.config = config
        self.nbones = nbones
        self.prediction = prediction
        recurrent_cell_box = torch.nn.ModuleList()
        if not prediction:
            self.seq_length_out = config.output_window_size
        else:
            self.seq_length_out = config.test_output_window
        for i in range(config.decoder_recurrent_steps):
            print("Prepare decoder for rec {}".format(str(i+1)))
            cells = torch.nn.ModuleList()
            for bone in range(nbones):
                cell = ST_LSTMCell(config)
                cells.append(cell)
            recurrent_cell_box.append(cells)
        self.recurrent_cell_box = recurrent_cell_box
        print("Prepare decoder for finished")

    def forward(self, hidden_states, cell_states, global_t_state, global_s_state, p):
        """

        :param hidden_states:  [batch, input_window_size-1, nbones, hidden_size]
        :param cell_states: [batch, input_window_size-1, nbones, hidden_size]
        :param global_t_state: [batch,  nbones, hidden_size]
        :param global_s_state: [batch, input_window_size-1, hidden_size]
        :param p: [batch, input_window_size-1, nbones, hidden_size]
        :return:
        """
        # h = torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones + 1, self.config.hidden_size, device=p.device)
        # h[:, 1:, 1:, :] = p
        # c_h = torch.zeros_like(h)
        # for i in range(self.config.decoder_recurrent_steps):
        #     if i == 0:
        #         h_t = hidden_states
        #         h_s = hidden_states
        #     elif i == 1:
        #         h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1)
        #         h_s = torch.cat((global_s_state.unsqueeze(2), hidden_states), dim=2)
        #
        #     h[:, 1:, 0, :] = torch.mean(torch.mean(h_s, dim=2), dim=1, keepdim=True)
        #     c_h[:, 1:, 0, :] = torch.mean(torch.mean(cell_states, dim=2), dim=1, keepdim=True)
        #     h[:, 0, 1:, :] = torch.mean(h_t, dim=1)
        #     c_h[:, 0, 1:, :] = torch.mean(cell_states, dim=1)
        #
        #     for frame in range(self.seq_length_out):
        #         for bone in range(self.nbones):
        #             cell = self.recurrent_cell_box[i][frame][bone]
        #             h[:, frame+1, bone+1, :], c_h[:, frame+1, bone+1, :] \
        #                 = cell(h[:, frame+1, bone+1, :].clone(), h[:, frame, bone+1, :].clone(),
        #                        h[:, frame+1, bone, :].clone(), c_h[:, frame, bone+1, :].clone(), c_h[:, frame+1, bone, :].clone())

        h_1 = torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones + 1, self.config.hidden_size, device=p.device)
        c_h_1 = torch.zeros_like(h_1)
        h_2 = torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones + 1, self.config.hidden_size, device=p.device)
        c_h_2 = torch.zeros_like(h_2)

        h_t = hidden_states
        h_s = hidden_states
        h_1[:, 1:, 0, :] = torch.mean(torch.mean(h_s, dim=2), dim=1, keepdim=True)
        c_h_1[:, 1:, 0, :] = torch.mean(torch.mean(cell_states, dim=2), dim=1, keepdim=True)
        h_1[:, 0, 1:, :] = torch.mean(h_t, dim=1)
        c_h_1[:, 0, 1:, :] = torch.mean(cell_states, dim=1)

        h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1)
        h_s = torch.cat((global_s_state.unsqueeze(2), hidden_states), dim=2)
        h_2[:, 1:, 0, :] = torch.mean(torch.mean(h_s, dim=2), dim=1, keepdim=True)
        c_h_2[:, 1:, 0, :] = torch.mean(torch.mean(cell_states, dim=2), dim=1, keepdim=True)
        h_2[:, 0, 1:, :] = torch.mean(h_t, dim=1)
        c_h_2[:, 0, 1:, :] = torch.mean(cell_states, dim=1)
        for frame in range(self.seq_length_out):
            for i in range(self.config.decoder_recurrent_steps):
                for bone in range(self.nbones):
                    cell = self.recurrent_cell_box[i][bone]
                    if i == 0:
                        if frame == 0:
                            h_1[:, frame+1, bone+1, :], c_h_1[:, frame+1, bone+1, :] \
                                = cell(p[:, 0, bone, :], h_1[:, frame, bone+1, :].clone(),
                                    h_1[:, frame+1, bone, :].clone(), c_h_1[:, frame, bone+1, :].clone(), c_h_1[:, frame+1, bone, :].clone())
                        else:
                            h_1[:, frame+1, bone+1, :], c_h_1[:, frame+1, bone+1, :] \
                                = cell(h_2[:, frame, bone+1, :].clone(), h_1[:, frame, bone+1, :].clone(),
                                    h_1[:, frame+1, bone, :].clone(), c_h_1[:, frame, bone+1, :].clone(), c_h_1[:, frame+1, bone, :].clone())
                    else:
                        h_2[:, frame + 1, bone + 1, :], c_h_2[:, frame + 1, bone + 1, :] \
                            = cell(h_1[:, frame + 1, bone + 1, :].clone(), h_2[:, frame, bone + 1, :].clone(),
                                   h_2[:, frame + 1, bone, :].clone(), c_h_2[:, frame, bone + 1, :].clone(),
                                   c_h_2[:, frame + 1, bone, :].clone())

        #print("Train finished")
        return h_2[:, 1:, 1:, :], c_h_2[:, 1:, 1:, :]


class ST_LSTMCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # input gate
        self.Ui = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wti = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wsi = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bi = torch.nn.Parameter(torch.randn(self.config.hidden_size))
        # space forget gate
        self.Us = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wts = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wss = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bs = torch.nn.Parameter(torch.randn(self.config.hidden_size))
        # time forget gate
        self.Ut = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtt = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wst = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bt = torch.nn.Parameter(torch.randn(self.config.hidden_size))
        # output gate
        self.Uo = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wto = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wso = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bo = torch.nn.Parameter(torch.randn(self.config.hidden_size))
        # c_hat gate
        self.Uc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wtc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.Wsc = torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size))
        self.bc = torch.nn.Parameter(torch.randn(self.config.hidden_size))

    def forward(self, x, h_t, h_s, c_t, c_s):

        i_n = torch.sigmoid(torch.matmul(x, self.Ui) + torch.matmul(h_t, self.Wti) + torch.matmul(h_s, self.Wsi) + self.bi)
        f_s_n = torch.sigmoid(torch.matmul(x, self.Us) + torch.matmul(h_t, self.Wts) + torch.matmul(h_s, self.Wss) + self.bs)
        f_t_n = torch.sigmoid(torch.matmul(x, self.Ut) + torch.matmul(h_t, self.Wtt) + torch.matmul(h_s, self.Wst) + self.bt)
        o_n = torch.sigmoid(torch.matmul(x, self.Uo) + torch.matmul(h_t, self.Wto) + torch.matmul(h_s, self.Wso) + self.bo)
        c_n = torch.tanh(torch.matmul(x, self.Uc) + torch.matmul(h_t, self.Wtc) + torch.matmul(h_s, self.Wsc) + self.bc)

        c_h = (i_n * c_n) + (f_t_n * c_t) + (f_s_n * c_s)
        h = o_n * torch.tanh(c_h)

        return h, c_h
