import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import numpy as np
import pickle
import einops

warnings.filterwarnings('ignore')


class Dataset_Temporal(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, np.float64(seq_x_mark), np.float64(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SpatioTemporalPKL(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='scs_lat_0to24_lon_105to121.pkl',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        with open(os.path.join(self.root_path, self.data_path), 'rb') as f:
            data_raw = pickle.load(f)
        '''
        df_raw.columns: ['metadata', 'sst', ...(other features), 'lat', 'lon', 'time']
        '''
        data_raw_keys = list(data_raw.keys())
        data_raw_keys.remove('metadata')
        data_raw_keys.remove('mask_ocean')
        data_raw_keys.remove('lat')
        data_raw_keys.remove('lon')
        data_raw_keys.remove(self.target)
        
        date = data_raw['time'] # array(['1981-09-01T11:00:00.000000', '1981-09-02T11:00:00.000000', '1981-09-03T11:00:00.000000', ..., '2022-12-29T11:00:00.000000', '2022-12-30T11:00:00.000000', '2022-12-31T11:00:00.000000'], dtype='datetime64[us]')
        # 设置date为pd.DataFrame date格式为'1981-09-01'
        date = pd.to_datetime(date).strftime('%Y-%m-%d')
        date = pd.DataFrame(date, columns=['date'])
        
        if self.features == 'M' or self.features == 'MS':
            # 堆叠其他变量数据(最后一维), target放在最后一维, 每个变量数据shape为(15097,96,64)
            data_raw = np.stack([data_raw[key] for key in data_raw_keys] + [data_raw[self.target]], axis=-1) # (15097,96,64,13)
        elif self.features == 'S':
            data_raw = data_raw[self.target].unsqueeze(-1) # (15097,96,64,1)

        self.mask = data_raw['mask_ocean']

        total_length = data_raw.shape[0]
        num_train = int(total_length * 0.7)
        num_test = int(total_length * 0.2)
        num_vali = total_length - num_train - num_test
        border1s = [0, num_train - self.seq_len, total_length - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, total_length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data_raw[border1s[0]:border2s[0]]
            train_data = einops.rearrange(train_data, 't h w c -> (t h w) c')
            self.scaler.fit(train_data)
            T, H, W, C = data_raw.shape
            data_raw = einops.rearrange(data_raw, 't h w c -> (t h w) c')
            data = self.scaler.transform(data_raw)
            data = einops.rearrange(data, '(t h w) c -> t h w c', t=T, h=H, w=W)
        else:
            data = data_raw

        df_stamp = date[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, np.float64(seq_x_mark), np.float64(seq_y_mark), self.mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        T, H, W, C = data.shape
        data = einops.rearrange(data, 't h w c -> (t h w) c')
        data = self.scaler.inverse_transform(data)
        data = einops.rearrange(data, '(t h w) c -> t h w c', t=T, h=H, w=W)
        return data


class Dataset_SpatioTemporal(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='scs_lat_0to24_lon_105to121_targetsst.npy',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_raw = np.load(os.path.join(self.root_path, self.data_path), mmap_mode='r')
        data_raw = np.nan_to_num(data_raw, nan=0.0)
        mask_ocean = np.load(os.path.join(self.root_path, 'scs_lat_0to24_lon_105to121_maskocean.npy'), mmap_mode='r')
        '''
        df_raw.columns: ['metadata', 'sst', ...(other features), 'lat', 'lon', 'time']
        '''
        date = pd.date_range(start='1981-09-01', end='2022-12-31', freq='D')
        date = pd.DataFrame(date, columns=['date'])
        
        if self.features == 'M' or self.features == 'MS':
            pass
        elif self.features == 'S':
            data_raw = data_raw[-1:] # (15097,96,64,1)

        self.mask = mask_ocean

        total_length = data_raw.shape[0]
        num_train = int(total_length * 0.7)
        num_test = int(total_length * 0.2)
        num_vali = total_length - num_train - num_test
        border1s = [0, num_train - self.seq_len, total_length - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, total_length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data_raw[border1s[0]:border2s[0]]
            train_data = einops.rearrange(train_data, 't h w c -> (t h w) c')
            self.scaler.fit(train_data)
            T, H, W, C = data_raw.shape
            data_raw = einops.rearrange(data_raw, 't h w c -> (t h w) c')
            data = self.scaler.transform(data_raw)
            data = einops.rearrange(data, '(t h w) c -> t h w c', t=T, h=H, w=W)
        else:
            data = data_raw

        df_stamp = date[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, np.float64(seq_x_mark), np.float64(seq_y_mark), self.mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        T, H, W, C = data.shape
        if C == 1:
            data = einops.repeat(data, 't h w c -> t h w (repeat c)', repeat=14)
        data = einops.rearrange(data, 't h w c -> (t h w) c')
        data = self.scaler.inverse_transform(data)
        data = einops.rearrange(data, '(t h w) c -> t h w c', t=T, h=H, w=W)
        if C == 1:
            data = data[..., -1]
        return data


class SharedStandardScaler:
    """管理 StandardScaler 的共享实例"""
    def __init__(self):
        self.scaler = None

    def fit(self, data):
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
        else:
            raise ValueError("StandardScaler has already been fit.")

    def transform(self, data):
        if self.scaler is not None:
            return self.scaler.transform(data)
        else:
            raise ValueError("StandardScaler has not been fit yet.")

    def inverse_transform(self, data):
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        else:
            raise ValueError("StandardScaler has not been fit yet.")


class Dataset_SpatioTemporalv2(Dataset):
    def __init__(self, shared_scaler: SharedStandardScaler, 
                 root_path, flag='train', size=None,
                 features='S', data_path='st_data.npy',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        if shared_scaler is None:
            raise ValueError("shared_scaler must be provided.")
        self.scaler = shared_scaler
        self.__read_data__()

    def __read_data__(self):

        mask_ocean = np.load(os.path.join(self.root_path, 'scs_lat_0to24_lon_105to121_maskocean.npy'), mmap_mode='r')
        self.mask = mask_ocean

        data_path_prefix, data_path_suffix = self.data_path.split('.')
        data_path_prefix = data_path_prefix + '_seq' + str(self.seq_len)
        new_data_path = data_path_prefix + '.' + data_path_suffix

        flag_data_path = os.path.join(self.root_path, self.flag, new_data_path)
        data_raw = np.load(flag_data_path, mmap_mode='r')

        date = pd.date_range(start='1981-09-01', end='2022-12-31', freq='D')
        date = pd.DataFrame(date, columns=['date'])
        
        if self.features == 'M' or self.features == 'MS':
            pass
        elif self.features == 'S':
            data_raw = data_raw[-1:] # (15097,96,64,1)

        total_length = 15097
        num_train = int(total_length * 0.7)
        num_test = int(total_length * 0.2)
        num_vali = total_length - num_train - num_test
        border1s = [0, num_train - self.seq_len, total_length - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, total_length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            if self.flag == 'train':
                train_data = data_raw
                train_data = einops.rearrange(train_data, 't h w c -> (t h w) c')
                self.scaler.fit(train_data)
            if self.flag == 'test' and self.scaler.scaler is None:
                train_data_path = os.path.join(self.root_path, 'train', new_data_path)
                train_data = np.load(train_data_path, mmap_mode='r')
                if self.features == 'S':
                    train_data = train_data[-1:] # (15097,96,64,1)
                train_data = einops.rearrange(train_data, 't h w c -> (t h w) c')
                self.scaler.fit(train_data)
            T, H, W, C = data_raw.shape
            data_raw = einops.rearrange(data_raw, 't h w c -> (t h w) c')
            data = self.scaler.transform(data_raw)
            data = einops.rearrange(data, '(t h w) c -> t h w c', t=T, h=H, w=W)
        else:
            data = data_raw

        df_stamp = date[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, np.float64(seq_x_mark), np.float64(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        T, H, W, C = data.shape
        if C == 1:
            data = einops.repeat(data, 't h w c -> t h w (repeat c)', repeat=14)
        data = einops.rearrange(data, 't h w c -> (t h w) c')
        data = self.scaler.inverse_transform(data)
        data = einops.rearrange(data, '(t h w) c -> t h w c', t=T, h=H, w=W)
        if C == 1:
            data = data[..., -1]
        return data