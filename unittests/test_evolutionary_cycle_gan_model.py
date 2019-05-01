import models.evolutionary_cycle_gan_model
import torch
import unittest
import math

class TestMutationCosts(unittest.TestCase):

    def setUp(self):
        self.vals = [0.9, 0.5, 0.01]
        self.test_tensor = torch.Tensor(self.vals)

    def test_minimax_cost(self):
        cost = models.evolutionary_cycle_gan_model.minimax_mutation_cost(self.test_tensor)
        self.assertAlmostEqual(float(cost), 1/6 * (math.log(1-0.9) + math.log(1- 0.5) + math.log(1 - 0.01)), 5)

    def test_heuristic_cost(self):
        cost = models.evolutionary_cycle_gan_model.heuristic_mutation_cost(self.test_tensor)
        self.assertAlmostEqual(float(cost), -1/6 * (math.log(0.9) + math.log(0.5) + math.log(0.01)), 5)

    def test_least_square_cost(self):
        cost = models.evolutionary_cycle_gan_model.least_square_mutation_cost(self.test_tensor)
        self.assertAlmostEqual(float(cost), 1/3 * sum([(i-1)**2 for i in self.vals]), 5)
