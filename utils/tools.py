import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_last_checkpoint(model, path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    
    def save_last_checkpoint(self, model, path):
        torch.save(model.state_dict(), path + '/' + 'last.pth')


class TrainTracking:
    def __init__(self, num_epochs, num_steps):
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.time_now = time.time()
        self.iter_count = 0

    def __call__(self, cur_step, cur_epoch, loss):
        self.iter_count += 1
        if (cur_step + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(cur_step + 1, cur_epoch + 1, loss.item()))
            speed = (time.time() - self.time_now) / self.iter_count
            left_time = speed * ((self.num_epochs - cur_epoch) * self.num_steps - cur_step)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            self.iter_count = 0
            self.time_now = time.time()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_ST(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization for spatio-temporal data
    true: [T, H, W] T is the time steps
    preds: [T, H, W] T is the time steps
    """
    T, H, W = true.shape
    fig, ax = plt.subplots(2, T, figsize=(T * 5, 5))
    for i in range(T):
        ax[0, i].imshow(true[i], cmap='jet')
        ax[0, i].set_title('GroundTruth')
        if preds is not None:
            ax[1, i].imshow(preds[i], cmap='jet')
            ax[1, i].set_title('Prediction')
    plt.savefig(name, bbox_inches='tight', dpi=200)


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def reserve_schedule_sampling_exp(itr, args):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (args.batch_size, args.seq_len - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (args.batch_size, args.pred_len - 1))
    true_token = (random_flip < eta)

    ones = np.ones((args.width // args.patch_size,
                    args.width // args.patch_size,
                    args.patch_size ** 2 * args.enc_in))
    zeros = np.zeros((args.width // args.patch_size,
                      args.width // args.patch_size,
                      args.patch_size ** 2 * args.enc_in))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.seq_len + args.pred_len - 2):
            if j < args.seq_len - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.seq_len - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.seq_len + args.pred_len - 2,
                                  args.width // args.patch_size,
                                  args.width // args.patch_size,
                                  args.patch_size ** 2 * args.enc_in))
    return real_input_flag


def schedule_sampling(eta, itr, args):
    zeros = np.zeros((args.batch_size,
                      args.pred_len - 1,
                      args.width // args.patch_size,
                      args.width // args.patch_size,
                      args.patch_size ** 2 * args.enc_in))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.pred_len - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.width // args.patch_size,
                    args.width // args.patch_size,
                    args.patch_size ** 2 * args.enc_in))
    zeros = np.zeros((args.width // args.patch_size,
                      args.width // args.patch_size,
                      args.patch_size ** 2 * args.enc_in))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.pred_len - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.pred_len - 1,
                                  args.width // args.patch_size,
                                  args.width // args.patch_size,
                                  args.patch_size ** 2 * args.enc_in))
    return eta, real_input_flag