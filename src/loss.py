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
    elif config.loss is 'kinematicslie':
        if config.dataset == 'Human':
            y = utils.prepare_loss(y, config.data_mean.shape[0], config.dim_to_ignore)
            prediction = utils.prepare_loss(prediction, config.data_mean.shape[0], config.dim_to_ignore)
        loss = kinematicslie_loss(prediction, y, bone, config)

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


def kinematicslie_loss(prediction, y, bone, config):
    """
    Our  loss function in the paper
    :param prediction:
    :param y:
    :param bone:
    :param config:
    :return:
    """

    bone_config = config.chain_loss_config
    chainlength = bone.shape[0]
    weights = torch.zeros(chainlength * 3, device=prediction.device)

    if len(bone_config) == 3:
        order = [1, 2, 0]
    elif len(bone_config) == 5:
        order = [0, 1, 3, 4, 2]
    count = 0.
    for i in order:
        indexs = bone_config[i]
        for j in range(len(indexs)):
            for k in range(j, len(indexs)):
                weights[indexs[j] * 3:indexs[j] * 3 + 3] = weights[indexs[j] * 3:indexs[j] * 3 + 3] + (len(indexs) - k) * bone[indexs[k]][0]

            if i != len(order) -1:
                count += (len(indexs) - j) * bone[indexs[j]][0]
            else:
                weights[indexs[j] * 3:indexs[j] * 3 + 3] += count

    #k = weights.max()
    weights = weights / weights.mean()
    loss = torch.sub(y, prediction) ** 2
    loss = torch.mean(loss, dim=[0, 1])
    loss = loss * weights
    loss = torch.mean(loss)
    return loss