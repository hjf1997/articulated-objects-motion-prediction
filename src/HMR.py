# implemented by Zhencheng Fan
# create time: 8/1/2019
import torch
import torch.nn as nn
import torch.nn.functional as F


class HMR(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_cell = torch.nn.ModuleList()
        # init encoder
        self.encoder_cell = HMR_EncoderCell(config)

        # init decoder
        self.decoder = LSTM_decoder(config)

        # 均匀分布初始化
        self.weights_in = torch.nn.Parameter(torch.empty(self.config.input_size, self.config.hidden_size).uniform_(-0.04, 0.04))
        self.bias_in = torch.nn.Parameter(torch.empty(self.config.hidden_size).uniform_(-0.04, 0.04))

    def forward(self, encoder_inputs, decoder_inputs, train):
        h = torch.matmul(encoder_inputs, self.weights_in) + self.bias_in
        h = F.dropout(h, p=self.config.keep_prob, training=train)

        c_h = torch.empty_like(h)
        c_h.copy_(h)
        c_h = F.dropout(c_h, p=self.config.keep_prob, training=train)

        hidden_states, cell_states, global_t_state = self.encoder_cell(h, c_h, train)
        prediction, _ = self.decoder(hidden_states, cell_states, global_t_state, decoder_inputs)
        return prediction


class HMR_EncoderCell(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = self.config.hidden_size
        # 纵向的层数
        self.rec_steps = self.config.encoder_recurrent_steps

        '''h update gates'''
        # forward forget gate,f_gate
        self.Uf = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Wlrf = torch.nn.Parameter(torch.empty(2 * self.hidden_size, self.hidden_size))
        self.Wf = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Zf = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bf = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.Uf, mean=0, std=1.)
        nn.init.normal_(self.Wlrf, mean=0, std=1.)
        nn.init.normal_(self.Wf, mean=0, std=1.)
        nn.init.normal_(self.Zf, mean=0, std=1.)
        nn.init.normal_(self.bf, mean=0, std=1.)

        # left forget gate
        self.Ul = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Wlrl = torch.nn.Parameter(torch.empty(2 * self.hidden_size, self.hidden_size))
        self.Wl = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Zl = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bl = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.Ul, mean=0, std=1.)
        nn.init.normal_(self.Wlrl, mean=0, std=1.)
        nn.init.normal_(self.Wl, mean=0, std=1.)
        nn.init.normal_(self.Zl, mean=0, std=1.)
        nn.init.normal_(self.bl, mean=0, std=1.)

        # right forget gate
        self.Ur = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Wlrr = torch.nn.Parameter(torch.empty(2 * self.hidden_size, self.hidden_size))
        self.Wr = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Zr = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.br = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.Ur, mean=0, std=1.)
        nn.init.normal_(self.Wlrr, mean=0, std=1.)
        nn.init.normal_(self.Wr, mean=0, std=1.)
        nn.init.normal_(self.Zr, mean=0, std=1.)
        nn.init.normal_(self.br, mean=0, std=1.)

        # forget gate for g
        self.Uq = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Wlrq = torch.nn.Parameter(torch.empty(2 * self.hidden_size, self.hidden_size))
        self.Wq = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Zq = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bq = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.Uq, mean=0, std=1.)
        nn.init.normal_(self.Wlrq, mean=0, std=1.)
        nn.init.normal_(self.Wq, mean=0, std=1.)
        nn.init.normal_(self.Zq, mean=0, std=1.)
        nn.init.normal_(self.bq, mean=0, std=1.)

        # input gate
        self.Ui = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Wlri = torch.nn.Parameter(torch.empty(2 * self.hidden_size, self.hidden_size))
        self.Wi = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Zi = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bi = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.Ui, mean=0, std=1.)
        nn.init.normal_(self.Wlri, mean=0, std=1.)
        nn.init.normal_(self.Wi, mean=0, std=1.)
        nn.init.normal_(self.Zi, mean=0, std=1.)
        nn.init.normal_(self.bi, mean=0, std=1.)

        # output gate
        self.Uo = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Wlro = torch.nn.Parameter(torch.empty(2 * self.hidden_size, self.hidden_size))
        self.Wo = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.Zo = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bo = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.Uo, mean=0, std=1.)
        nn.init.normal_(self.Wlro, mean=0, std=1.)
        nn.init.normal_(self.Wo, mean=0, std=1.)
        nn.init.normal_(self.Zo, mean=0, std=1.)
        nn.init.normal_(self.bo, mean=0, std=1.)

        '''g update gates'''

        # forget gates for h
        self.g_Wf = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.g_Zf = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.g_bf = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.g_Wf, mean=0, std=1.)
        nn.init.normal_(self.g_Zf, mean=0, std=1.)
        nn.init.normal_(self.g_bf, mean=0, std=1.)

        # forget gate for g
        self.g_Wg = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.g_Zg = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.g_bg = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.g_Wg, mean=0, std=1.)
        nn.init.normal_(self.g_Zg, mean=0, std=1.)
        nn.init.normal_(self.g_bg, mean=0, std=1.)

        # output gate
        self.g_Wo = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.g_Zo = torch.nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.g_bo = torch.nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.g_Wo, mean=0, std=1.)
        nn.init.normal_(self.g_Zo, mean=0, std=1.)
        nn.init.normal_(self.g_bo, mean=0, std=1.)

    def forward(self, h, c_h, train):
        # record shape of the batch
        shape = h.shape
        # initial g
        g = torch.mean(h, dim=1)
        c_g = torch.mean(c_h, dim=1)

        final_hidden_states = []
        final_cell_states = []
        final_global_states = []

        padding = torch.zeros_like(h[:, 0:self.config.context_window, :], device=h.device)

        for i in range(self.rec_steps):
            '''update g'''
            g_tilde = torch.mean(h, dim=1)
            # forget gates for g
            # h这里是[batch,frames,hidden_size]，需要reshape成[-1,hidden_size]
            reshaped_h = h.view(-1, self.hidden_size)
            reshaped_g = g.unsqueeze(dim=1).repeat(1, shape[1], 1).view(-1, self.hidden_size)

            g_f_n = torch.sigmoid(torch.matmul(reshaped_g, self.g_Zf) + torch.matmul(reshaped_h, self.g_Wf) + self.g_bf)
            g_g_n = torch.sigmoid(torch.matmul(g, self.g_Zg) + torch.matmul(g_tilde, self.g_Wg) + self.g_bg)
            g_o_n = torch.sigmoid(torch.matmul(g, self.g_Zo) + torch.matmul(g_tilde, self.g_Wo) + self.g_bo)

            # reshape the gates
            # 后面*乘，不是matmul,这里需要reshape
            reshaped_g_f_n = g_f_n.view(-1, shape[1], self.hidden_size)
            reshaped_g_g_n = g_g_n.view(-1, 1, self.hidden_size)

            # update c_g and g
            c_g_n = torch.sum(reshaped_g_f_n * c_h, dim=1) + torch.squeeze(reshaped_g_g_n, dim=1) * c_g
            g_n = g_o_n * torch.tanh(c_g_n)

            '''update h'''
            # [-1, config.input_window_size-1, config.hidden_size]
            # get states before/after
            h_before = [self.get_hidden_states_before(h, step + 1, padding).view(-1, self.hidden_size) for step in
                        range(self.config.context_window)]
            h_before = self.sum_together(h_before)
            h_after = [self.get_hidden_states_after(h, step + 1, padding).view(-1, self.hidden_size) for step in
                       range(self.config.context_window)]
            h_after = self.sum_together(h_after)

            # get cells before/after
            c_h_before = [self.get_hidden_states_before(c_h, step + 1, padding).view(-1, self.hidden_size) for step in
                          range(self.config.context_window)]
            c_h_before = self.sum_together(c_h_before)
            c_h_after = [self.get_hidden_states_after(c_h, step + 1, padding).view(-1, self.hidden_size) for step in
                         range(self.config.context_window)]
            c_h_after = self.sum_together(c_h_after)

            # reshape for matmul
            reshaped_h = h.view(-1, self.hidden_size)
            reshaped_c_h = c_h.view(-1, self.hidden_size)
            reshaped_g = (g.unsqueeze(dim=1)).repeat(1, shape[1], 1).view(-1, self.hidden_size)
            reshaped_c_g = (c_g.unsqueeze(dim=1)).repeat(1, shape[1], 1).view(-1, self.hidden_size)

            # concat before and after hidden states
            h_before_after = torch.cat((h_before, h_after), dim=1)

            # forget gates for h
            f_n = torch.sigmoid(torch.matmul(reshaped_h, self.Uf) + torch.matmul(h_before_after, self.Wlrf) +
                                torch.matmul(reshaped_h, self.Wf) + torch.matmul(reshaped_g, self.Zf) + self.bf)
            l_n = torch.sigmoid(torch.matmul(reshaped_h, self.Ul) + torch.matmul(h_before_after, self.Wlrl) +
                                torch.matmul(reshaped_h, self.Wl) + torch.matmul(reshaped_g, self.Zl) + self.bl)
            r_n = torch.sigmoid(torch.matmul(reshaped_h, self.Ur) + torch.matmul(h_before_after, self.Wlrr) +
                                torch.matmul(reshaped_h, self.Wr) + torch.matmul(reshaped_g, self.Zr) + self.br)
            q_n = torch.sigmoid(torch.matmul(reshaped_h, self.Uq) + torch.matmul(h_before_after, self.Wlrq) +
                                torch.matmul(reshaped_h,self.Wq) + torch.matmul(reshaped_g, self.Zq) + self.bq)
            i_n = torch.sigmoid(torch.matmul(reshaped_h, self.Ui) + torch.matmul(h_before_after, self.Wlri) +
                                torch.matmul(reshaped_h,self.Wi) + torch.matmul(reshaped_g, self.Zi) + self.bi)
            o_n = torch.sigmoid(torch.matmul(reshaped_h, self.Uo) + torch.matmul(h_before_after, self.Wlro) +
                                torch.matmul(reshaped_h,self.Wi) + torch.matmul(reshaped_g, self.Zo) + self.bo)

            # 形状待调试
            c_h_n = (l_n * c_h_before) + (r_n * c_h_after) + (f_n * reshaped_c_h) +\
                    (q_n * reshaped_c_g) + (i_n * reshaped_c_h)
            h_n = o_n * torch.tanh(c_h_n)

            # update states
            h = h_n.view(-1, shape[1], self.hidden_size)
            c_h = c_h_n.view(-1, shape[1], self.hidden_size)

            g = g_n
            c_g = c_g_n

            final_hidden_states.append(h)
            final_cell_states.append(c_h)
            final_global_states.append(g)

            h = F.dropout(h, p=self.config.keep_prob, training=train)
            c_h = F.dropout(c_h, p=self.config.keep_prob, training=train)

        hidden_states = final_hidden_states[-1]
        cell_states = final_cell_states[-1]
        global_states = final_global_states[-1].view(-1, 1, self.hidden_size)

        return hidden_states, cell_states, global_states

    def get_hidden_states_before(self, hidden_states, step, padding):
        displaced_hidden_states = hidden_states[:, :-step, :]
        return torch.cat((padding, displaced_hidden_states), dim=1)

    def sum_together(self, l):
        combined_state = None
        for tensor in l:
            if combined_state == None:
                combined_state = tensor
            else:
                combined_state = combined_state + tensor
        return combined_state

    def get_hidden_states_after(self, hidden_states, step, padding):
        displaced_hidden_states = hidden_states[:, step:, :]
        return torch.cat((displaced_hidden_states, padding), dim=1)


class LSTM_decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_length_out = config.output_window_size
        self.lstm = nn.ModuleList()
        self.weights_out = torch.nn.Parameter(
            torch.empty(self.config.hidden_size, self.config.input_size).uniform_(-0.04, 0.04))
        self.bias_out = torch.nn.Parameter(torch.empty(config.input_size).uniform_(-0.04, 0.04))
        for i in range(config.decoder_recurrent_steps):
            if i == 0:
                self.lstm.append(nn.LSTMCell(config.input_size, config.hidden_size))
            else:
                self.lstm.append(nn.LSTMCell(config.hidden_size, config.hidden_size))

    def forward(self, hidden_states, cell_states, global_t_state, p):
        # hidden_states、cell_states:batch,frames,hidden_size
        # global_t_state:batch,1,hidden_size
        # p:batch,frame,hidden_size
        # define decoder hidden states and cell states
        h = []
        c_h = []
        pre = torch.zeros([hidden_states.shape[0], self.seq_length_out, self.config.input_size], device=p.device)
        for i in range(self.config.decoder_recurrent_steps):
            # 双层LSTM
            h.append(torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.config.hidden_size, device=p.device))
            c_h.append(torch.zeros_like(h[i]))
            # feed init hidden states and cell states into h and c_h
            if i == 0:
                h_t = hidden_states
            elif i == 1:
                h_t = torch.cat((global_t_state, hidden_states), dim=1)  # 这个是Hg
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

                h[i][:, frame + 1, :], c_h[i][:, frame + 1, :] = cell(input, (h[i][:, frame, :].clone(), c_h[i][:, frame, :].clone()))
            pre[:, frame, :] = torch.matmul(h[-1][:, frame + 1, :].clone(), self.weights_out) + \
                               self.bias_out + input_first

        pre_c = c_h[-1][:, 1:, :]
        return pre, pre_c
