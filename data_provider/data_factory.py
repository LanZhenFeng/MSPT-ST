from data_provider.data_loader import Dataset_Temporal, Dataset_SpatioTemporal, Dataset_SpatioTemporalv2
from torch.utils.data import DataLoader

data_dict = {
    'temporal': Dataset_Temporal,
    'spatiotemporal': Dataset_SpatioTemporalv2,
}


def data_provider(args, shared_scaler, flag, test_batch_size=1, pin_memory=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = test_batch_size  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        shared_scaler=shared_scaler,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=True)
    return data_set, data_loader
