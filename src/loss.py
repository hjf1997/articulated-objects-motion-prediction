# implemented by JunfengHu
# create time: 7/25/2019

import torch
import utils


def loss(prediction, y, bone, config):

    if config.loss is 'l2':
        loss = l2_loss(prediction, y)
    elif config.loss is 'lie':
        if config.dataset == 'Human':
            y = utils.prepare_loss(y, config.data_mean.shape[0], config.dim_to_ignore)
            prediction = utils.prepare_loss(prediction, config.data_mean.shape[0], config.dim_to_ignore)
        loss = linearizedlie_loss(prediction, y, bone, config)

    return loss


def l2_loss(prediction, y):

    loss = torch.sub(y, prediction) ** 2
    loss = torch.mean(loss)
    return loss


def linearizedlie_loss(prediction, y, bone, config):
    """
    Lie loss
    :param prediction:
    :param y:
    :param bone:
    :param config:
    :return:
    """
    if config.dataset is not 'Human':
        chainlength = bone.shape[0] - 1
        weights = torch.zeros(chainlength * 3, device=prediction.device)
        for j in range(chainlength):
            for i in range(j, chainlength):
                weights[j*3:j*3+3] = weights[j*3] + (chainlength - i) * bone[i + 1][0]
    else:
        chainlength = bone.shape[0]
        weights = torch.zeros(chainlength * 3, device=prediction.device)
        for j in range(chainlength):
            for i in range(j, chainlength):
                weights[j*3:j*3+3] = weights[j*3] + (chainlength - i) * bone[i][0]
    weights = weights / weights.max()
    loss = torch.sub(y, prediction) ** 2
    loss = torch.mean(loss, dim=[0, 1])
    loss = loss * weights
    loss = torch.mean(loss)

    return loss

