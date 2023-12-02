import unittest
import app
import json

class FlaskTest(unittest.TestCase):

    # Ensure predict/svm route behaves correctly
    def test_predict_svm(self):
        tester = app.app.test_client(self)
        response = tester.post('/predict/svm', json={"image": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})  # Replace with an appropriate image array
        statuscode = response.status_code
        self.assertEqual(statuscode, 200)
        self.assertTrue("y_predicted" in response.get_json())

    # Ensure predict/lr route behaves correctly
    def test_predict_lr(self):
        tester = app.app.test_client(self)
        response = tester.post('/predict/lr', json={"image": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})  # Replace with an appropriate image array
        statuscode = response.status_code
        self.assertEqual(statuscode, 200)
        self.assertTrue("y_predicted" in response.get_json())

    # Ensure predict/tree route behaves correctly
    def test_predict_tree(self):
        tester = app.app.test_client(self)
        response = tester.post('/predict/tree', json={"image": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})  # Replace with an appropriate image array
        statuscode = response.status_code
        self.assertEqual(statuscode, 200)
        self.assertTrue("y_predicted" in response.get_json())

if __name__ == '__main__':
    unittest.main()
