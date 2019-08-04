# implemented by JunfengHu
# create time: 7/25/2019

import torch
import numpy as np


def l2_loss(prediction, y):

    loss = torch.sub(y, prediction) ** 2
    loss = torch.mean(loss)
    return loss

def linearizedlie_loss(prediction, y, bone):

    chainlength = bone.shape[0] - 1
    weights = torch.zeros(chainlength * 3, device=prediction.device)
    for j in range(chainlength):
        for i in range(j, chainlength):

            weights[j*3:j*3+3] = weights[j*3] + (chainlength - i) * bone[i + 1][0]
    weights = weights / weights.max()
    loss = torch.sub(y, prediction) ** 2
    loss = torch.mean(loss, dim=[0, 1])
    loss = loss * weights
    loss = torch.mean(loss)

    return loss

# def lie_loss(prediction, y, config):
#
#     joint_pred = forward_kinematics(prediction, config)
#     joint_label = forward_kinematics(y, config)
#     loss = torch.mean(torch.sub(joint_pred, joint_label) ** 2)

