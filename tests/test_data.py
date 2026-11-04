import unittest
import json

class TestDataIntegrity(unittest.TestCase):
    def test_treatments_json_is_valid(self):
        # اختبار بسيط للتأكد إن ملف العلاجات سليم
        with open('treatments.json', 'r') as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

if __name__ == '__main__':
    unittest.main()