import argparse


class Parameter:
    def __init__(self):
        self.args = self.set_args()

    def set_args(self):
        self.parser = argparse.ArgumentParser(description='Physically boosted cooperative learning framework')

        # Global parameters
        self.parser.add_argument('--local_rank', type=int, default=-1)
        self.parser.add_argument('--seed', type=int, default=42, help='random seed')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        self.parser.add_argument('--mode', type=str, default='formal', help='continue of formal')
        self.parser.add_argument('--model_name', type=str, default='83MAE_train/')

        # Data parameters
        self.parser.add_argument('--frame_length', type=int, default=15, help='length of one sequence')
        self.parser.add_argument('--save_dir', type=str, default='../data/checkpoints/', help='directory to save the models')
        self.parser.add_argument('--results_dir', type=str, default='../results/', help='directory to save the results')
        self.parser.add_argument('--data_root', type=str, default='/ayt1/anyitong/TurbMAE/2D/PBCL+81_MAE/data/dataset_2500/')
        # self.parser.add_argument('--data_root', type=str, default='../data/small_dataset/')

        # Model parameters
        self.parser.add_argument('--n_feats', type=int, default=16)
        self.parser.add_argument('--dropout', type=int, default=0.2, help='dropout for all')
        self.parser.add_argument('--neighboring_frames', type=int, default=2, help='use of neighboring frames')

        # Training parameters
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--start_epoch', type=int, default=1, help='first epoch number')
        self.parser.add_argument('--end_epoch', type=int, default=50, help='last epoch number')

        args, _ = self.parser.parse_known_args()

        return args
