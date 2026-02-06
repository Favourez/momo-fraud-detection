import unittest
from src.custom_algorithms import merge_sort, binary_search_threshold

class TestCustomAlgorithms(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.data = [
            {'amount': 50, 'risk_score': 0.2},
            {'amount': 100, 'risk_score': 0.8},
            {'amount': 75, 'risk_score': 0.5}
        ]

    def test_merge_sort(self):
        sorted_data = merge_sort(self.data, 'risk_score')
        self.assertEqual([d['risk_score'] for d in sorted_data], [0.2, 0.5, 0.8])

    def test_binary_search_threshold(self):
        sorted_data = merge_sort(self.data, 'risk_score')
        high_risk = binary_search_threshold(sorted_data, 'risk_score', 0.5)
        self.assertEqual([d['risk_score'] for d in high_risk], [0.5, 0.8])

if __name__ == '__main__':
    unittest.main()
