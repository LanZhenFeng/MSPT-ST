import os
import torch
from models import MSPT, ConvLSTM, E3DLSTM, MAU, MIM, PhyDNet, PredRNN, PredRNNPP, PredRNNv2, SwinLSTM_B, SwinLSTM_D, SimVP, TAU, PredFormer
from models import MSPSTT_I, MSPSTT_IR, MSPSTT_S, MSPSTT_SR, MSPSTT_T, MSPSTT_TR, MSPSTT_U, MSPSTT_UR, MSPSTT_IFD, MSPSTT_IFDA, MSPSTT_IFDB, MSPSTT_IFDC

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MSPT': MSPT,
            'MSPSTT_I': MSPSTT_I,
            'MSPSTT_IR': MSPSTT_IR,
            'MSPSTT_IFD': MSPSTT_IFD,
            'MSPSTT_IFDA': MSPSTT_IFDA,
            'MSPSTT_IFDB': MSPSTT_IFDB,
            'MSPSTT_IFDC': MSPSTT_IFDC,
            'MSPSTT_S': MSPSTT_S,
            'MSPSTT_SR': MSPSTT_SR,
            'MSPSTT_T': MSPSTT_T,
            'MSPSTT_TR': MSPSTT_TR,
            'MSPSTT_U': MSPSTT_U,
            'MSPSTT_UR': MSPSTT_UR,
            'ConvLSTM': ConvLSTM,
            'E3DLSTM': E3DLSTM,
            'MAU': MAU,
            'MIM': MIM,
            'PhyDNet': PhyDNet,
            'PredRNN': PredRNN,
            'PredRNNPP': PredRNNPP,
            'PredRNNv2': PredRNNv2,
            'SwinLSTM_B': SwinLSTM_B,
            'SwinLSTM_D': SwinLSTM_D,
            'SimVP': SimVP,
            'TAU': TAU,
            'PredFormer': PredFormer
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass