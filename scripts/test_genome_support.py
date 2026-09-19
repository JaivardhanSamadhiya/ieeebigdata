"""Synthetic checks of the fixed gate's train-only dependency contract."""
import unittest
import numpy as np
from genome_support_diagnostic import support_gate, mean_or_none

class SupportGateTests(unittest.TestCase):
    def setUp(self):
        self.sim = np.array([[1,.8,.6,.9,.2],[.8,1,.7,.1,.3],[.6,.7,1,.4,.5],
                             [.9,.1,.4,1,.99],[.2,.3,.5,.99,1]])
        self.train = np.array([0,1,2]); self.test = np.array([3,4])

    def test_self_exclusion(self):
        scores, threshold, accepted = support_gate(self.sim,self.train,self.test)
        self.assertAlmostEqual(threshold,.71)
        np.testing.assert_allclose(scores,[.9,.5])
        np.testing.assert_array_equal(accepted,[True,False])

    def test_query_query_and_diagonal_do_not_affect_gate(self):
        expected = support_gate(self.sim,self.train,self.test)
        changed = self.sim.copy()
        changed[np.ix_(self.test,self.test)] = 0
        np.fill_diagonal(changed,100)
        actual = support_gate(changed,self.train,self.test)
        for a,b in zip(expected,actual): np.testing.assert_equal(a,b)

    def test_test_similarity_cannot_change_threshold(self):
        _, before, _ = support_gate(self.sim,self.train,self.test)
        changed = self.sim.copy()
        changed[np.ix_(self.test,self.train)] = 0
        _, after, accepted = support_gate(changed,self.train,self.test)
        self.assertEqual(before,after)
        self.assertFalse(accepted.any())
        self.assertIsNone(mean_or_none(np.array([])))

if __name__ == '__main__': unittest.main()
