from .base_options import BaseOptions


class EvaluationOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--evaluation-model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152'])
        parser.add_argument('--evaluation-checkpoint', type=str, default='evaluation1')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--validation-accuracy-goal', type=float, default=95.0, help='The training will stop once it reaches this validation accuracy')
        parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs to train model')
        parser.add_argument('--phase', type=str, default='train')

        self.isTrain= False
        return parser
