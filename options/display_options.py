from .base_options import BaseOptions


class DisplayOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--genA_path', type=str, help='path to generator A')
        parser.add_argument('--genB_path', type=str, help='path to generator A')
        self.isTrain= False
        return parser
