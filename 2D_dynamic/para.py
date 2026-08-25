import argparse


class Parameter:
    def __init__(self):
        self.args = self.set_args()

    def set_args(self):
        self.parser = argparse.ArgumentParser(description='Physically boosted cooperative learning framework')

        # Global parameters
        self.parser.add_argument('--seed', type=int, default=40, help='random seed')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--mode', type=str, default='formal', help='continue of formal')
        self.parser.add_argument('--model_name', type=str, default='')

        # Data parameters
        self.parser.add_argument('--frame_length', type=int, default=30, help='length of one sequence')
        self.parser.add_argument('--save_dir', type=str, default='./model/experiment/', help='directory to save the models')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='directory to save the results')
        self.parser.add_argument('--data_root', type=str, default='/ayt1/anyitong/TurbMAE/2D_dynamic/dataset_2500', help='the path of dataset')

        # Model parameters
        self.parser.add_argument('--n_blocks', type=int, default=10, help='# of blocks in middle part of the model')
        self.parser.add_argument('--n_features', type=int, default=16)
        self.parser.add_argument('--future_frames', type=int, default=2)
        self.parser.add_argument('--past_frames', type=int, default=2)
        self.parser.add_argument('--dropout', type=int, default=0.2, help='dropout for all')
        self.parser.add_argument('--activation', type=str, default='gelu', help='activation function')
        

        # Training parameters
        self.parser.add_argument('--start_epoch', type=int, default=1, help='first epoch number')
        self.parser.add_argument('--end_epoch', type=int, default=300, help='last epoch number')

        args = self.parser.parse_args()

        return args
