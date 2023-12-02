import unittest
from joblib import load
from sklearn.linear_model import LogisticRegression

class TestLogisticRegressionModel(unittest.TestCase):
    rollno = "M23CSA020" 

    def test_model_type(self):
        for solver in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']:
            model = load(f"{self.rollno}_lr_{solver}.joblib")
            self.assertIsInstance(model, LogisticRegression)

    def test_solver_name(self):
        for solver in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']:
            model = load(f"{self.rollno}_lr_{solver}.joblib")
            self.assertEqual(model.get_params()['solver'], solver)

if __name__ == '__main__':
    unittest.main()
