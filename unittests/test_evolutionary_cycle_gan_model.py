import models.evolutionary_cycle_gan_model
import torch
import unittest
import math
import torch.nn.functional as F

class TestMutationCosts(unittest.TestCase):

    def setUp(self):
        vals = [-4, 0, 4]
        # self.vals  = [F.sigmoid(v) for v in vals]
        self.test_tensor = torch.Tensor(vals)
        self.sigmoid_vals = [float(i) for i in list(torch.sigmoid(self.test_tensor))]
        # print(F.sigmoid(self.test_tensor))
        print(self.sigmoid_vals)

    def test_minimax_cost(self):
        cost = models.evolutionary_cycle_gan_model.minimax_mutation_cost(self.test_tensor)
        self.assertAlmostEqual(float(cost), 1/6 * (math.log(1-0.01798621) + math.log(1- 0.5) + math.log(1 - 0.98201379)), 5)

    def test_heuristic_cost(self):
        cost = models.evolutionary_cycle_gan_model.heuristic_mutation_cost(self.test_tensor)
        self.assertAlmostEqual(float(cost), -1/6 * (math.log(0.01798621) + math.log(0.5) + math.log(0.98201379)), 5)

    def test_least_square_cost(self):
        cost = models.evolutionary_cycle_gan_model.least_square_mutation_cost(self.test_tensor)
        self.assertAlmostEqual(float(cost), 1/3 * sum([(i-1)**2 for i in list(self.sigmoid_vals)]), 5)
