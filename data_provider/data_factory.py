from data_provider.data_loader import Dataset_ETT_hour, \
    Dataset_ETT_minute, Dataset_Custom, Dataset_M4, Dataset_EFP_long, Dataset_EFP_h2, Dataset_EFP_each, Dataset_EFP_RAG
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'EFP_long': Dataset_EFP_long,
    'EFP_h2': Dataset_EFP_h2,
    'EFP_each': Dataset_EFP_each,
    'EFP_RAG': Dataset_EFP_RAG
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            target=args.target,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            target=args.target,
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        # num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
