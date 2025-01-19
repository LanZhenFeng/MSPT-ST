import cv2
import numpy as np
import torch
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr

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
def MSE_ST(pred, true, spatial_norm=False):
    # pred [B, T, H, W, C], true [B, T, H, W, C]
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()

def RMSE_ST(pred, true, spatial_norm=False):
    # pred [B, T, H, W, C], true [B, T, H, W, C]
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())

def MAE_ST(pred, true, spatial_norm=False):
    # pred [B, T, H, W, C], true [B, T, H, W, C]
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()

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


def PSNR_ST(pred, true):
    psnr = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            # psnr += PSNR(pred[i, j], true[i, j])
            psnr += cal_psnr(true[i, j].swapaxes(0, 2), pred[i, j].swapaxes(0, 2), data_range=true[i, j].max() - true[i, j].min())
    return psnr / (pred.shape[0] * pred.shape[1])


def SSIM(pred, true):
    # pred [B, T, H, W, C], true [B, T, H, W, C], mask [H, W]
    ssim = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            ssim += cal_ssim(pred[i, j].swapaxes(0, 2), true[i, j].swapaxes(0, 2), data_range=true[i, j].max() - true[i, j].min(), channel_axis=0)
    return ssim / (pred.shape[0] * pred.shape[1])


def metric_st(pred, true, mask):
    pred = pred * mask
    true = true * mask
    print(pred.shape, true.shape)
    mse = MSE_ST(pred, true, False)
    mae = MAE_ST(pred, true, False)
    rmse = RMSE_ST(pred, true, False)
    psnr = PSNR_ST(pred, true)
    ssim = SSIM(pred, true)
    return mse, mae, rmse, psnr, ssim