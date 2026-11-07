import unittest
from app import app

class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
    def test_home(self):
        self.assertEqual(self.client.get('/').status_code, 200)
    def test_prediction(self):
        response = self.client.post('/api/predict', json={'pH': 6.8, 'N': 95, 'P': 55, 'K': 120, 'OC': 1.8, 'moisture': 42})
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json['health_class'], ('Poor', 'Moderate', 'Healthy'))
    def test_invalid(self):
        response = self.client.post('/api/predict', json={'pH': 50})
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
