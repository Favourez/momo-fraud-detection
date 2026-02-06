# Assuming 'type' has been encoded: 0=PAYMENT, 1=TRANSFER, 2=CASH_OUT, 3=DEBIT, 4=CASH_IN
class FraudRiskScorer:
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_risk(self):
        self.df['risk_score'] = 0
        
        # Step 1: Transaction type (TRANSFER or CASH_OUT is riskier)
        self.df.loc[self.df['type'].isin([1,2]), 'risk_score'] += 1
        
        # Step 2: High transaction amount
        amount_threshold = self.df['amount'].mean()
        self.df.loc[self.df['amount'] > amount_threshold, 'risk_score'] += 1
        
        # Step 3: Sender balance zero after transaction
        self.df.loc[self.df['newbalanceOrig'] == 0, 'risk_score'] += 1
        
        # Step 4: Flag high-risk transactions
        self.df['isHighRisk'] = (self.df['risk_score'] >= 2).astype(int)
        
        return self.df

"""
The FraudRiskScorer class in the risk_algorithm.py is a custom rule-based risk scoring system. It does not train a model, but instead calculates a risk score for each transaction using simple rules, which can then be used as a feature in your ML model.

Starts with a column risk_score = 0 for all transactions.

Adds points for high-risk characteristics:

**Rules implemented:**

Transaction type

If the transaction is TRANSFER or CASH_OUT, it’s considered higher risk → add 1 point.

High transaction amount

If amount is greater than the mean transaction amount → add 1 point.

Sender balance zero after transaction

If newbalanceOrig == 0 → add 1 point.
"""