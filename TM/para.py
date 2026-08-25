import argparse

class Parameter:
    def __init__(self):
        self.args = self.set_args()

    def set_args(self):
        self.parser = argparse.ArgumentParser(description='Turbulent image analysis')

        # global parameters
        self.parser.add_argument('--mode', type=str, default='continue', help='demo, formal, or continue')
        self.parser.add_argument('--resume_file', type=str, default='checkpoints_92_attn/DATUM_68.pth', help='the path of checkpoint file for resume')

        # data parameters
        self.parser.add_argument('--data_root', type=str, default='./gopro_ds_ori', help='the path of dataset')
        self.parser.add_argument('--frame_length', type=int, default=12, help='frames in a sequence')

        # training parameters
        self.parser.add_argument('--end_epoch', type=int, default=100, help='last epoch number')
        args, _ = self.parser.parse_known_args()

        return args