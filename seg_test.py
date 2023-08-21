# unittest for seg.py
import unittest
from seg import find_corr_candidate_pts


class TestCPtsPair(unittest.TestCase):
    def test_find_candidate_pts(self):
        test_cases = [
            {
                'xs': [0, 0, 0, 0, 0],
                'ys': [1, 2, 3, 4, 5],
                'c_pts': [(0, 10), (2, 10), (10, 2), (0, 1)],
                'expected': [(0, 10), (2, 10)]
            }, {
                'xs': [1, 2, 3, 4, 5],
                'ys': [0, 0, 0, 0, 0],
                'c_pts': [(10, 0), (10, 2), (2, 10)],
                'expected': [(10, 0), (10, 2)]
            }, {
                'xs': [2, 3, 5, 5],
                'ys': [4, 3, 2, 1],
                'c_pts': [(2, 6), (4, 2)],
                'expected': []
            }
        ]
        for case in test_cases:
            ret = find_corr_candidate_pts(
                list(zip(case['xs'], case['ys'])), case['c_pts'])

            assert sorted([tuple(x[0])
                          for x in ret]) == sorted(case['expected'])
