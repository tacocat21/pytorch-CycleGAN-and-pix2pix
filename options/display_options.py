from .base_options import BaseOptions


class DisplayOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        parser.add_argument('--genA_load_path', type=str, help='path to generator A')
        parser.add_argument('--genB_load_path', type=str, help='path to generator B')
        self.isTrain= False
        return parser
