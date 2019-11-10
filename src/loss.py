# implemented by JunfengHu
# create time: 7/25/2019

import torch
import utils


def loss(prediction, y, bone, config):

    if config.loss == 'l2':
        loss = l2_loss(prediction, y)
    elif config.loss == 'weightlie':
        if config.dataset == 'Human':
            y = utils.prepare_loss(y, config.data_mean.shape[0], config.dim_to_ignore)
            prediction = utils.prepare_loss(prediction, config.data_mean.shape[0], config.dim_to_ignore)
        loss = weightlie_loss(prediction, y, bone, config)
    elif config.loss == 'HMRlie':
        if config.dataset == 'Human':
            y = utils.prepare_loss(y, config.data_mean.shape[0], config.dim_to_ignore)
            prediction = utils.prepare_loss(prediction, config.data_mean.shape[0], config.dim_to_ignore)
        loss = HMRlie_loss(prediction, y, bone, config)

    return loss


def l2_loss(prediction, y):

    loss = torch.sub(y, prediction) ** 2
    loss = torch.mean(loss)
    return loss

def HMRlie_loss(prediction, y, bone, config):
    """

    :param prediction:
    :param y:
    :param bone:
    :param config:
    :return:
    """

    chainlength = bone.shape[0]
    weights = torch.zeros(chainlength * 3, device=prediction.device)
    for j in range(chainlength):
            weights[j*3:j*3+3] = (chainlength - j) * bone[j][0]

    weights = weights / weights.max()
    loss = torch.sub(y, prediction) ** 2
    loss = torch.mean(loss, dim=[0, 1])
    loss = loss * weights
    loss = torch.mean(loss)

    return loss

def weightlie_loss(prediction, y, bone, config):
    """
    weightlielie loss
    :param prediction:
    :param y:
    :param bone:
    :param config:
    :return:
    """

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
