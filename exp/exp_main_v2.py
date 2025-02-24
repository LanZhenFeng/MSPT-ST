from data_provider.data_factory import data_provider
from data_provider.data_loader import SharedStandardScaler
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, TrainTracking, CurriculumLearning, adjust_learning_rate, visual, visual_st
from utils.metrics import metric, metric_st
from utils.loss import MaskedMSELoss, MaskedMAELoss, MaskedMSEMAELoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    r"""
    Expersiment Main Class for SST Spatial-Temporal Dataset with Mask (Land or Ocean)
    """
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        assert args.data == 'spatiotemporalv2', 'Data type should be "spatiotemporalv2"'
        self.shared_scaler = SharedStandardScaler()
        self.curriculum_learning = CurriculumLearning(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, test_batch_size=1):
        data_set, data_loader = data_provider(self.args, flag, test_batch_size, shared_scaler=self.shared_scaler)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_scheduler(self, optimizer, lr, train_steps, train_epochs):
        if self.args.lradj == 'exp':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.args.lradj == 'constant':
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 ** epoch)
        elif self.args.lradj == 'half':
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
        elif self.args.lradj == 'cyclic':
            return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=lr, step_size_up=train_steps, step_size_down=train_steps, mode='triangular2', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=False)
        elif self.args.lradj == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=0)
        elif self.args.lradj == 'cosine_wr':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=0)
        elif self.args.lradj == 'reduce':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        elif self.args.lradj == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=train_steps, epochs=train_epochs, pct_start=0.3)
        elif self.args.lradj == 'none':
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 ** epoch)

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = MaskedMSELoss(self.args.auxiliary_loss_weight)
        elif self.args.loss == 'MAE':
            criterion = MaskedMAELoss(self.args.auxiliary_loss_weight)
        elif self.args.loss == 'MSE+MAE':
            criterion = MaskedMSEMAELoss(self.args.auxiliary_loss_weight)
        return criterion

    def _makedirs(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.results_save_path):
            os.makedirs(self.results_save_path)
        if not os.path.exists(self.test_results_save_path):
            os.makedirs(self.test_results_save_path)

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_true, **kwargs):
        if self.args.use_amp:
            with torch.amp.autocast('cuda'):
                outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_true, **kwargs)
        else:
            outputs, aux_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_true, **kwargs)
        if aux_loss is not None and self.args.use_multi_gpu and self.args.use_gpu:
            aux_loss = aux_loss.mean()
        return outputs, aux_loss
    
    def vali(self, vali_data, vali_loader, criterion):
        self.mask = torch.from_numpy(vali_data.mask).to(self.device)
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                if self.args.model_type == 'rb':
                    dec_inp = batch_y.to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, ..., :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, ..., :], dec_inp], dim=1).float().to(self.device)

                # mask_true
                mask_true = self.curriculum_learning.get_mask_true_on_testing(batch_x.shape[0])
                if mask_true is not None:
                    mask_true = mask_true.float().to(self.device)

                # encoder - decoder
                outputs, aux_loss = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_true)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, ..., f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, ..., f_dim:].to(self.device)

                loss = criterion((outputs, aux_loss), batch_y, self.mask)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test', test_batch_size=self.args.batch_size)

        self.mask = torch.from_numpy(train_data.mask).to(self.device)

        self.model_save_path = os.path.join(self.args.model_save_path, setting)
        self.results_save_path = os.path.join(self.args.results_save_path, setting)
        self.test_results_save_path = os.path.join(self.args.test_results_save_path, setting)
        
        early_stopping = EarlyStopping(watch_epoch=self.args.watch_epoch, patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        train_steps = len(train_loader)
        
        scheduler = self._select_scheduler(model_optim, self.args.learning_rate, train_steps, self.args.train_epochs)
        print(f"Using {self.args.lradj} learning rate adjustment")

        criterion = self._select_criterion()
        # test_criterion = self._select_criterion()
        test_criterion = MaskedMSELoss()
        
        self._makedirs()

        # save args
        args_dict = vars(self.args)
        with open(os.path.join(self.model_save_path, 'config.txt'), 'w') as f:
            for k, v in args_dict.items():
                f.write(str(k) + ' = ' + str(v) + '\n')
        
        with open(os.path.join(self.results_save_path, 'config.txt'), 'w') as f:
            for k, v in args_dict.items():
                f.write(str(k) + ' = ' + str(v) + '\n')
        
        with open(os.path.join(self.test_results_save_path, 'config.txt'), 'w') as f:
            for k, v in args_dict.items():
                f.write(str(k) + ' = ' + str(v) + '\n')

        if self.args.use_amp:
            scaler = torch.amp.GradScaler('cuda')

        # reset memory stats
        print("Resetting memory stats...")
        torch.cuda.reset_peak_memory_stats(self.device)

        print("Starting training")
        for epoch in range(self.args.train_epochs):
            train_track = TrainTracking(self.args.train_epochs, train_steps)
            train_loss = []

            num_updates = epoch * train_steps

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                if self.args.model_type == 'rb':
                    dec_inp = batch_y.to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, ..., :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, ..., :], dec_inp], dim=1).float().to(self.device)

                # mask_true
                mask_true = self.curriculum_learning.get_mask_true_on_training(num_updates, batch_x.shape[0])
                if mask_true is not None:
                    mask_true = mask_true.float().to(self.device)

                # encoder - decoder
                outputs, aux_loss = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_true)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, ..., f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, ..., f_dim:].to(self.device)
                
                loss = criterion((outputs, aux_loss), batch_y, self.mask)
                train_loss.append(loss.item())

                train_track(i, epoch, loss)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                if self.args.lradj == "onecycle":
                    scheduler.step()

                num_updates += 1
        
            if self.args.lradj != "onecycle":
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, test_criterion)
            test_loss = self.vali(test_data, test_loader, test_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.model_save_path)
            print("Adjusting learning rate to: {:.7f}".format(scheduler.get_last_lr()[0]))
            if self.device.type == "cuda":
                print(f"memory footprint: {torch.cuda.max_memory_allocated(self.device) / 1024 ** 2:.3f} MB")
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self, setting, load_weight=True):
        test_batch_size = self.args.batch_size if self.args.model == 'MIM' else 1
        test_data, test_loader = self._get_data(flag='test', test_batch_size=test_batch_size)
        self.mask = torch.from_numpy(test_data.mask).to(self.device)
        if load_weight:
            print('loading supervised model weight')
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_save_path + setting, 'checkpoint.pth'), map_location=self.device))
            test_results_save_path = self.args.test_results_save_path + setting + '/'
            results_save_path = self.args.results_save_path + setting + '/'

        if not os.path.exists(test_results_save_path):
            os.makedirs(test_results_save_path)

        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
    
        preds = []
        trues = []

        # reset memory stats
        print("Resetting memory stats...")
        torch.cuda.reset_peak_memory_stats(self.device)

        print("starting testing")
    
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                if self.args.model_type == 'rb':
                    dec_inp = batch_y.to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, ..., :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, ..., :], dec_inp], dim=1).float().to(self.device)

                # mask_true
                mask_true = self.curriculum_learning.get_mask_true_on_testing(batch_x.shape[0])
                if mask_true is not None:
                    mask_true = mask_true.float().to(self.device)

                # encoder - decoder
                outputs, _ = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_true)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, ..., f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, ..., f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    if outputs.shape[0] > 1:
                        for i in range(outputs.shape[0]):
                            outputs[i] = test_data.inverse_transform(outputs[i]).reshape(shape[1:])
                            batch_y[i] = test_data.inverse_transform(batch_y[i]).reshape(shape[1:])
                    else:
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                
                outputs = outputs[..., f_dim:]
                batch_y = batch_y[..., f_dim:]

                pred = outputs.astype(np.float16)
                true = batch_y.astype(np.float16)
                preds.append(pred)
                trues.append(true)
                if i % 100 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    # if test_data.scale and self.args.inverse:
                    #     shape = input.shape
                    #     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    # gt = np.concatenate((input[0, -365:, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, -365:, -1], pred[0, :, -1]), axis=0)
                    gt = true[0, :, ..., -1]
                    pd = pred[0, :, ..., -1]
                    visual_st(gt, pd, os.path.join(test_results_save_path, str(i) + '.pdf'))
        
        if self.device.type == "cuda":
            print(f"memory footprint: {torch.cuda.max_memory_allocated(self.device) / 1024 ** 2:.3f} MB")

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-4], preds.shape[-3], preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, preds.shape[-4], preds.shape[-3], trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mask = self.mask.detach().cpu().numpy().astype(np.bool_)
        mask = mask[None, None, :, :, None]
        print(mask.shape)

        preds *= mask
        trues *= mask

        np.save(results_save_path + 'pred.npy', preds)
        np.save(results_save_path + 'true.npy', trues)

    def cal_metrics(self, setting):
        results_save_path = self.args.results_save_path + setting + '/'
        preds = np.load(results_save_path + 'pred.npy')
        trues = np.load(results_save_path + 'true.npy')
        
        mse, mae, rmse, pnsr, ssim = metric_st(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, pnsr:{}, ssim:{}'.format(mse, mae, rmse, pnsr, ssim))
        f = open("result_sstp_st_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, pnsr:{}, ssim:{}'.format(mse, mae, rmse, pnsr, ssim))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(results_save_path + 'metrics.npy', np.array([mse, mae, rmse, pnsr, ssim]))

    def get_paramandflops(self):
        from thop import profile
        from thop import clever_format
        input = torch.randn(1, self.args.seq_len, self.args.height, self.args.width, self.args.enc_in).to(self.device)
        input_mark = torch.randn(1, self.args.seq_len, 3).to(self.device)
        dec_inp = torch.randn(1, self.args.label_len+self.args.pred_len, self.args.height, self.args.width, self.args.dec_in).to(self.device)
        dec_inp_mark = torch.randn(1, self.args.label_len+self.args.pred_len, 3).to(self.device)

        mask_true = self.curriculum_learning.get_mask_true_on_testing(input.shape[0])
        if mask_true is not None:
            mask_true = mask_true.float().to(self.device)

        flops, params = profile(self.model, inputs=(input, input_mark, dec_inp, dec_inp_mark, mask_true))

        flops_1 = f"{flops / 10**9:.3f}G"
        params_1 = f"{params / 10**6:.3f}M"
        print("-" * 50)
        print(f"FLOPS: {flops_1}, Params: {params_1}")
        print("-" * 50)
        flops_2, params_2 = clever_format([flops, params], "%.3f")
        print(f"FLOPS: {flops_2}, Params: {params_2}")
        print("-" * 50)

    def get_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    # def get_layer_output(self, inp, layers=None, unwrap=False):
    #     """
    #     Args:
    #         inp: can be numpy array, torch tensor or dataloader
    #     """
    #     self.model.eval()
    #     device = next(self.model.parameters()).device
    #     if isinstance(inp, np.ndarray): inp = torch.Tensor(inp).to(device)
    #     if isinstance(inp, torch.Tensor): inp = inp.to(device)
        
    #     return get_layer_output(inp, model=self.model, layers=layers, unwrap=unwrap)

    # def get_layer_output(self, inp, model, layers=None, unwrap=False):
    #     """
    #     layers is a list of module names
    #     """
    #     orig_model = model
        
    #     if unwrap: model = unwrap_model(model)
    #     if not layers: layers = list(dict(model.named_children()).keys())
    #     if not isinstance(layers, list): layers = [layers]

    #     activation = {}
    #     def getActivation(name):
    #         # the hook signature
    #         def hook(model, input, output):
    #             activation[name] = output.detach().cpu().numpy()
    #         return hook

    #     # register forward hooks on the layers of choice    
    #     h_list = [getattr(model, layer).register_forward_hook(getActivation(layer)) for layer in layers]
        
    #     model.eval()
    #     out = orig_model(inp)    
    #     for h in h_list: h.remove()
    #     return activation