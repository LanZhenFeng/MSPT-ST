import cv2
import numpy as np
import torch
from sklearn.metrics import r2_score

'''
Metrics for evaluation of temporal forecasting models
'''

# 相对平方误差
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

# 相关系数
def CORR(pred, true):
    # pred [B, T, C], true [B, T, C]
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0) #  [T, C]
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0)) # [T, C]
    return (u / d).mean(-1) # [T]

# 平均绝对误差
def MAE(pred, true):
    return np.mean(np.abs(pred - true))

# 均方误差
def MSE(pred, true):
    return np.mean((pred - true) ** 2)

# 均方根误差
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

# 平均绝对百分比误差
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

# 平均平方百分比误差
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def ACC(pred, true):
    return 1 - np.mean(np.abs(pred - true) / true)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rse = RSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)
    r2score = r2_score(pred.squeeze(-1), true.squeeze(-1))
    acc = ACC(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, r2score, acc

'''
Metrics for evaluation of spatio-temporal forecasting models
'''
def MSE_ST(pred, true):
    # pred [B, T, H, W, C], true [B, T, H, W, C]
    return np.mean((pred - true) ** 2)

def Masked_MSE_ST(pred, true, mask):
    # pred [B, T, H, W, C], true [B, T, H, W, C], mask [H, W]
    mask = mask[None, None, :, :, None]
    return np.sum((pred - true) ** 2 * mask) / np.sum(mask)

def RMSE_ST(pred, true):
    # pred [B, T, H, W, C], true [B, T, H, W, C]
    return np.sqrt(MSE_ST(pred, true))

def Masked_RMSE_ST(pred, true, mask):
    # pred [B, T, H, W, C], true [B, T, H, W, C], mask [H, W]
    mask = mask[None, None, :, :, None]
    return np.sqrt(Masked_MSE_ST(pred, true, mask))

def MAE_ST(pred, true):
    # pred [B, T, H, W, C], true [B, T, H, W, C]
    return np.mean(np.abs(pred - true))

def Masked_MAE_ST(pred, true, mask):
    # pred [B, T, H, W, C], true [B, T, H, W, C], mask [H, W]
    mask = mask[None, None, :, :, None]
    return np.sum(np.abs(pred - true) * mask) / np.sum(mask)


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def metric_st(pred, true, mask):
    masked_mse = Masked_MSE_ST(pred, true, mask)
    masked_rmse = Masked_RMSE_ST(pred, true, mask)
    masked_mae = Masked_MAE_ST(pred, true, mask)
    psnr = PSNR(pred, true)
    ssim = SSIM(pred, true)
    return masked_mse, masked_mae, masked_rmse, psnr, ssim