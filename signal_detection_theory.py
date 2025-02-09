import unittest 
import numpy as np # use pip3 install numpy in terminal
import scipy # use pip3 install scipy in terminal
from scipy.stats import norm

class SignalDetection: 
    def __init__(self, hits, misses, falseAlarms, correctRejections): # code gotten from ChatGPT
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def hit_rate(self):
        return self.hits / (self.hits + self.misses) if (self. hits + self.misses) > 0 else 0
    
    def false_alarm_rate(self):
        return self.falseAlarms / (self.falseAlarms + self.correctRejections) if (self.falseAlarms + self.correctRejections) > 0 else 0
    
    def d_prime(self):
        hit_rate = self.hit_rate()
        false_alarm_rate = self.false_alarm_rate()

        hit_rate = min(max(hit_rate, 1e-10), 1 - 1e-10)
        false_alarm_rate = min(max(false_alarm_rate, 1e-10), 1 - 1e-10)

        return norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
    
    def criterion(self):
        hit_rate = self.hit_rate()
        false_alarm_rate = self.false_alarm_rate()

        hit_rate = min(max(hit_rate, 1e-10), 1 - 1e-10)
        false_alarm_rate = min(max(false_alarm_rate, 1e-10), 1 - 1e-10)

        return -0.5 * (norm.ppf(hit_rate) + norm.ppf(false_alarm_rate))


class TestSignalDetection(unittest.TestCase):
    
    def test_init(self):
        sd = SignalDetection(10, 5, 8, 12)
        self.assertEqual(sd.hits, 10)
        self.assertEqual(sd.misses, 5)
        self.assertEqual(sd.falseAlarms, 8)
        self.assertEqual(sd.correctRejections, 12)
    
    def test_d_prime(self):
        sd = SignalDetection(15, 5, 10, 10)
        expected = norm.ppf(sd.hit_rate()) - norm.ppf(sd.false_alarm_rate())
        self.assertAlmostEqual(sd.d_prime(), expected, places=6)
    
    def test_criterion(self):
        sd = SignalDetection(10, 10, 5, 15)
        expected = -0.5 * (norm.ppf(sd.hit_rate()) + norm.ppf(sd.false_alarm_rate()))
        self.assertAlmostEqual(sd.criterion(), expected, places=6)

if __name__ == '__main__':
    unittest.main()

