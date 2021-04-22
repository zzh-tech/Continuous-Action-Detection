import argparse


class Parameter:
    def __init__(self):
        self.args = self.extract_args()

    def extract_args(self):
        self.parser = argparse.ArgumentParser()
        # global parameters
        self.parser.add_argument('--seed', type=int, default=39, help='random seed')
        self.parser.add_argument('--threads', type=int, default=2, help='# of threads for dataloader')
        self.parser.add_argument('--cpu', action='store_true', help='run on CPU, not recommended')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='# of GPUs to use')
        self.parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_file', type=str, default='', help='the path of checkpoint file')
        self.parser.add_argument('--test_state', action='store_true', help='show if the state is to do the test')

        # data parameters
        self.parser.add_argument('--num_sensors', type=int, default=1, help='# of IMUs used')
        self.parser.add_argument('--data_root', type=str, default='./data/', help='the path of dataset')
        self.parser.add_argument('--dataset', type=str, default='NSE', help='the dataset used in the project')
        self.parser.add_argument('--save_dir', type=str, default='./experiment/',
                                 help='directory to save logs of experiments')
        self.parser.add_argument('--data_length', type=int, default=688,
                                 help='max length of original data samples')  # 674

        # model parameters
        self.parser.add_argument('--model', type=str, default='MSSTCN', help='type of model to construct')
        self.parser.add_argument('--num_channels', type=int, default=6, help='# of signal channels')
        self.parser.add_argument('--num_feats', type=int, default=256, help='# of channel features')
        self.parser.add_argument('--num_layers', type=int, default=10, help='# of layers in one stage')
        self.parser.add_argument('--num_stages', type=int, default=3, help='# of stages in one stream')
        self.parser.add_argument('--num_classes', type=int, default=8, help='# of action classes')
        self.parser.add_argument('--dropout', type=float, default=0.3, help='possibility of dropout')

        # loss parameters
        self.parser.add_argument('--loss', type=str, default='Mixed', help='type of loss function')

        # optimizer parameters
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='method of optimization')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--milestones', type=int, nargs='*', default=[5, 10])
        self.parser.add_argument('--decay_gamma', type=float, default=0.5, help='decay rate')

        # training parameters
        self.parser.add_argument('--start_epoch', type=int, default=1, help='first epoch number')
        self.parser.add_argument('--end_epoch', type=int, default=15, help='last epoch number')
        self.parser.add_argument('--trainer_mode', type=str, default='dp',
                                 help='trainer mode: distributed data parallel (ddp) or data parallel (dp)')

        # test parameters
        self.parser.add_argument('--test_only', action='store_true', help='test using pretrained model')
        self.parser.add_argument('--test_checkpoint', type=str,
                                 default='./experiment/MSSTCN_CMHAD_Transiti/checkpoint.pth.tar',
                                 help='pretained checkpoint')

        args, _ = self.parser.parse_known_args()
        if args.dataset == 'CMHAD_Transition':
            args.num_classes = 8
        elif args.dataset == 'CMHAD_Gesture':
            args.num_classes = 6
        else:
            raise NotImplementedError

        return args
