from .base_options import BaseOptions


class EvaluationGANOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--evaluation-model-filename', type=str, help='Evaluation model filename')
        parser.add_argument('--evaluation-checkpoint', type=str, default='evaluation1')
        parser.add_argument('--evolutionary', action='store_true', help='Train the model using evolutionary algorithm')
        parser.add_argument('--genA_load_path', type=str, help='path to generator A')
        parser.add_argument('--genB_load_path', type=str, help='path to generator B')
        self.isTrain= False
        return parser
