import models.evolutionary_cycle_gan_model
import torch
import unittest

class TestHeuristicCosts(unittest.TestCase):

    def setUp(self):
        self.test_tensor = torch.Tensor([1., 0.5, 0.01])

    def test_minimax_cost(self):
        cost = models.evolutionary_cycle_gan_model.minimax_mutation_cost(self.test_tensor)
        self.assertEqual(cost, )

    def test_heuristic_cost(self):
        cost = models.evolutionary_cycle_gan_model.heuristic_mutation_cost(self.test_tensor)
        self.assertEqual(cost, )

    def test_least_square_cost(self):
        cost = models.evolutionary_cycle_gan_model.least_square_mutation_cost(self.test_tensor)
        self.assertEqual(cost, )
