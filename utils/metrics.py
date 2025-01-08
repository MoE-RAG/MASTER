import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true):
    # 确保分母不为零
    denominator = np.abs(pred) + np.abs(true)
    denominator = np.where(denominator == 0, 1e-8, denominator)  # 避免分母为零

    # 计算 SMAPE
    smape_value = 2.0 * np.mean(np.abs(pred - true) / denominator) * 100  # 乘以100得到百分比
    return smape_value


def MASE(pred, true):
    # 计算分子：预测误差的绝对值平均
    numerator = np.mean(np.abs(true - pred))

    # 计算分母：naive预测误差的绝对值平均
    naive_error = np.mean(np.abs(true[1:] - true[:-1]))

    # 避免分母为零
    if naive_error == 0:
        return np.inf  # 或者返回一个合理的默认值

    # 计算 MASE
    mase = numerator / naive_error
    return mase


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    mase = MASE(pred, true)

    return mae, mse, rmse, mape, mspe, smape, mase
