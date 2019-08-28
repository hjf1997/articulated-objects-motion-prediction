import torch
import torch.nn as nn


class ERD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # init encoder
        self.encoder_cell = ERD_EncoderCell(config)

        # init decoder
        self.decoder = ERD_Decoder(config)

    def forward(self, encoder_inputs, decoder_inputs, train):
        enc_state = self.encoder_cell(encoder_inputs, train)
        prediction, _ = self.decoder(enc_state, decoder_inputs)
        return prediction


class ERD_EncoderCell(nn.Module):
    # Encoder结构
    # 输入为[input_window_size-1][batch,bones*3]
    # 每个[batch,bones*3]输入到一个包含500个单元的全连接层，激活函数为relu，
    def __init__(self, config):
        super().__init__()
        self.config = config
        nbones = config.nbones
        self.config.hidden_size = 1000
        self.hidden_size = [self.config.hidden_size, self.config.hidden_size]
        self.number_of_layers = len(self.hidden_size)
        # 全连接层，输入为[batch,bones*3]
        self.fc_layer = nn.Sequential(nn.Linear(nbones * 3, 500), nn.ReLU(True))

        # LSTM
        self.lstm = nn.ModuleList()
        self.lstm.append(nn.LSTMCell(500, self.config.hidden_size))
        self.lstm.append(nn.LSTMCell(self.config.hidden_size, self.config.hidden_size))

    def forward(self, enc_in, train):
        # 纵向LSTM要进行dropout，需要把输入[batch,frames-1,bones*3]搞成[frames-1,batch,bones*3]
        enc_in = enc_in.permute(1, 0, 2)

        # 输入到全连接层,输出为(batch,500)
        fc = [self.fc_layer(torch.squeeze(enc_in[i, :, :])) for i in range(enc_in.shape[0])]

        # 输入到LSTM层
        h = []
        c_h = []

        for i in range(self.number_of_layers):
            h.append(torch.zeros(len(fc), fc[0].shape[0], self.config.hidden_size, device=enc_in.device))
            c_h.append(torch.zeros_like(h[i]))

        for frame in range(enc_in.shape[0]):
            for i in range(self.number_of_layers):
                cell = self.lstm[i]
                if i == 0:
                    input = torch.squeeze(fc[i])
                else:
                    # 这里的input需要进行dropout
                    input = torch.dropout(torch.squeeze(h[i - 1][frame, :, :]), p=self.config.keep_prob, train=train)
                # 第i层第frame+1帧的h值们
                if (frame == 0):
                    h[i][frame, :, :], c_h[i][frame, :, :] = cell(input)
                else:
                    h[i][frame, :, :], c_h[i][frame, :, :] = cell(input, (torch.squeeze(h[i][frame - 1, :, :].clone())
                                                                          , torch.squeeze(
                        c_h[i][frame - 1, :, :]).clone()))
        enc_state = [(torch.squeeze(h[0][len(fc) - 1, :, :]), torch.squeeze(c_h[0][len(fc) - 1, :, :])),
                     (torch.squeeze(h[1][len(fc) - 1, :, :]), torch.squeeze(c_h[1][len(fc) - 1, :, :]))]
        return enc_state


class ERD_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_length_out = config.output_window_size
        self.lstm = nn.ModuleList()
        self.config.hidden_size = 1000
        self.nbones = config.nbones

        # LSTM
        self.lstm = nn.ModuleList()
        self.lstm.append(nn.LSTMCell(500, self.config.hidden_size))
        self.lstm.append(nn.LSTMCell(self.config.hidden_size, self.config.hidden_size))

        # fc_in_1
        self.fc_in_1 = nn.Sequential(nn.Linear(self.nbones * 3, 500), nn.ReLU(True))
        # fc_in_2
        self.fc_in_2 = nn.Sequential(nn.Linear(500, 500), nn.ReLU(False))
        # fc1
        self.fc1 = nn.Sequential(nn.Linear(self.config.hidden_size, 500), nn.ReLU(True))
        # fc2
        self.fc2 = nn.Sequential(nn.Linear(500, 100), nn.ReLU(True))
        # out
        self.fc_out = nn.Sequential(nn.Linear(100, self.nbones * 3), nn.ReLU(False))

    def forward(self, enc_state, p):
        h = []
        c_h = []
        # 由于下一帧的input不是h，而h又要用，因此要一个output
        output = torch.empty(self.seq_length_out, enc_state[0][0].shape[0], self.nbones * 3)

        for i in range(self.config.decoder_recurrent_steps):
            # 双层LSTM
            # h[i].shape=[output_window_size+1,batch,hidden_size]
            h.append(
                torch.zeros(self.seq_length_out, enc_state[0][0].shape[0], self.config.hidden_size, device=p.device))
            c_h.append(torch.zeros_like(h[i]))
            # feed init hidden states and cell states into h and c_h
            # h_t是即h,c_h就是c_h,这里的enc_state[0][0]即单个h或者c_h形状是[batch,hidden_size]
            if i == 0:
                h_t = enc_state[0][0].clone()
            elif i == 1:
                h_t = enc_state[1][0].clone()
            else:
                print('The max decoder num is 2!')

            # 表示第i层的第0个h和c_h
            h[i][0, :, :] = h_t
            c_h[i][0, :, :] = enc_state[i][1]

        for frame in range(self.seq_length_out):
            # 第0帧的时候，输入是p，形状应该是[batch,hidden_size]
            if (frame == 0):
                fc_in_1 = self.fc_in_1(torch.squeeze(p).clone())
            # 如果是第二帧的位置，输入就是上一帧第二层的output的值
            else:
                fc_in_1 = self.fc_in_1(torch.squeeze(output[frame - 1, :, :]).clone())
            fc_in_2 = self.fc_in_2(fc_in_1)

            input = fc_in_2
            # 由两个全连接层处理完开始进入双层LSTM层，输入是全连接层的输出fc_in_2，形状[batch,500]
            for i in range(self.config.decoder_recurrent_steps):
                cell = self.lstm[i]
                if (i == 0):
                    h[i][frame, :, :], c_h[i][frame, :, :] = cell(input, (
                    h[i][frame - 1, :, :].clone(), c_h[i][frame - 1, :, :].clone()))
                else:
                    h[i][frame, :, :], c_h[i][frame, :, :] = cell(h[i - 1][frame, :, :].clone(), (
                    h[i][frame - 1, :, :].clone(), c_h[i][frame - 1, :, :].clone()))

            # 后三个全连接层
            fc1 = self.fc1(torch.squeeze(h[-1][frame, :, :]).clone())
            fc2 = self.fc2(fc1)
            output[frame, :, :] = self.fc_out(fc2)

        pre = output
        pre_c = c_h[-1][:, 1:, :]
        pre = pre.permute(1, 0, 2)
        return pre, pre_c
