import argparse
import os
import torch
# from exp.exp_main_rb import Exp_Main
# from exp.exp_main_rf import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Multi-Scale Periodicity Transformer (MSPT)')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='MSPT',
                        help='model name, options: [MSPT, MSPT-ST]')
    parser.add_argument('--model_type', type=str, default='rb', help='model type, options: [rb, rf]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='spatiotemporal', help='dataset type, options: [temporal, spatiotemporal]')
    parser.add_argument('--root_path', type=str, default='./data/Area/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='scs_lat_0to24_lon_105to121_targetsst.npy', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--model_save_path', type=str, default='/root/autodl-tmp/checkpoints/', help='path  to save model')
    parser.add_argument('--results_save_path', type=str, default='/root/autodl-tmp/results/', help='path to save results')
    parser.add_argument('--test_results_save_path', type=str, default='/root/autodl-tmp/test_results/', help='path to save test results')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--individual', action='store_true', help='channel independence', default=False)
    parser.add_argument('--position_wise', action='store_true', help='position wise', default=False)
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--height', type=int, default=96, help='size of height dimension for SpatioTemporal data')
    parser.add_argument('--width', type=int, default=64, help='size of width dimension for SpatioTemporal data')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size along height and width dimension')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--curriculum_learning_strategy', type=str, default='none',
                        help='curriculum learning strategy, options:[rss, ss, s, none]')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--auxiliary_loss_weight', type=float, default=0.1, help='auxiliary loss weight')
    parser.add_argument('--output_attention', action='store_true', help='output attention weights', default=False)
    parser.add_argument('--stride_scale', type=int, default=1, help='stride scale')
    parser.add_argument('--pre_norm', action='store_true', help='pre norm', default=False)
    parser.add_argument('--is_parallel', action='store_true', help='parallel spatio-temporal attention or not', default=False)
    parser.add_argument('--use_conv', action='store_true', help='use conv in gate layer', default=False)
    parser.add_argument('--use_linear', action='store_true', help='use linear in gate layer', default=False)
    parser.add_argument('--is_rotary', action='store_true', help='use rotary position encoding', default=False)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--watch_epoch', type=int, default=1, help='early stopping watch epoch')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='onecycle', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    # reverse scheduled sampling
    parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
    parser.add_argument('--r_sampling_step_1', type=float, default=25000)
    parser.add_argument('--r_sampling_step_2', type=int, default=50000)
    parser.add_argument('--r_exp_alpha', type=int, default=5000)
    # scheduled sampling
    parser.add_argument('--scheduled_sampling', type=int, default=1)
    parser.add_argument('--sampling_stop_iter', type=int, default=50000)
    parser.add_argument('--sampling_start_value', type=float, default=1.0)
    parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # special setting for SimVP
    args.model_print = args.model
    if args.model.startswith('SimVP_'):
        args.attn_type = args.model.split('_')[1]
        args.model = 'SimVP'
        attn_types = ['gSTA', 'ConvMixer', 'ConvNeXt', 'HorNet', 'MLPMixer', 'MogaNet', 'Poolformer', 'Swin', 'Uniformer', 'VAN', 'ViT']
        assert args.attn_type in attn_types, 'model type should be in {}'.format(attn_types)

    # special setting for PredFormer
    if args.model.startswith('PredFormer_'):
        args.attn_type = args.model.split('_')[1]
        args.model = 'PredFormer'
        attn_types = ['Full', 'FacTS', 'FacST', 'BinaryTS', 'BinaryST', 'TripletTST', 'TripletSTS', 'QuadrupletTSST', 'QuadrupletSTTS']
        assert args.attn_type in attn_types, 'attn type should be in {}'.format(attn_types)

    print('Args in experiment:')
    print(args)

    if args.data == 'spatiotemporal':
        from exp.exp_main_v1 import Exp_Main

    elif args.data == 'spatiotemporalv2':
        from exp.exp_main_v2 import Exp_Main
    
    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_h{}w{}_ps{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_cls{}_{}_{}'.format(
                args.model_id,
                args.model_print,
                args.model_type,
                args.data,
                args.features,
                args.height,
                args.width,
                args.patch_size,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.embed,
                args.curriculum_learning_strategy,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, load_weight=True)
            torch.cuda.empty_cache()

            print('>>>>>>>calculate metrics : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.cal_metrics(setting)

            print('>>>>>>>get params and flops : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.get_paramandflops()
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_h{}w{}_ps{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_cls{}_{}_{}'.format(
            args.model_id,
            args.model_print,
            args.model_type,
            args.data,
            args.features,
            args.height,
            args.width,
            args.patch_size,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.embed,
            args.curriculum_learning_strategy,
            args.des, ii)
        
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, load_weight=True) 
        torch.cuda.empty_cache()

        print('>>>>>>>calculate metrics : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.cal_metrics(setting)
        
        print('>>>>>>>get params and flops : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.get_paramandflops()
        torch.cuda.empty_cache()

     