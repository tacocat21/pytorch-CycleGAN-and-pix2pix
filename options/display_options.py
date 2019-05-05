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

        parser.add_argument('--fakeA_image_path', type=str, default='./fake_A.png', help='image path to grid of generated A images eg -> ./image_path.png')
        parser.add_argument('--fakeB_image_path', type=str, default='./fake_B.png', help='image path to grid of generated B images eg -> ./image_path.png')

        parser.add_argument('--realA_image_path', type=str, default='./real_A.png', help='image path to grid of real A images eg -> ./image_path.png')
        parser.add_argument('--realB_image_path', type=str, default='./real_B.png', help='image path to grid of real B images eg -> ./image_path.png')

        parser.add_argument('--grid_size', type=int, default=8, help='Sq. Root. of number of generated images in grid')

        self.isTrain= False
        return parser
