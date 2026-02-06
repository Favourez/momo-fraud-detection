import unittest
import pandas as pd
from src.risk_algorithm import FraudRiskScorer

class TestFraudRiskScorer(unittest.TestCase):
    def setUp(self):
        # Small sample DataFrame
        self.df = pd.DataFrame({
            'step': [1, 1],
            'type': ['PAYMENT', 'TRANSFER'],
            'amount': [100, 500],
            'oldbalanceOrg': [1000, 2000],
            'newbalanceOrig': [900, 1500],
            'nameOrig': ['C1', 'C2'],
            'nameDest': ['M1', 'M2']
        })

    def test_calculate_risk(self):
        scorer = FraudRiskScorer(self.df)
        df_risk = scorer.calculate_risk()
        self.assertIn('risk_score', df_risk.columns)
        self.assertTrue((df_risk['risk_score'] >= 0).all())

if __name__ == '__main__':
    unittest.main()
