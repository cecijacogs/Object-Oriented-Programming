import unittest 
import numpy as np # use pip3 install numpy in terminal
import scipy # use pip3 install scipy in terminal
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdt.SignalDetection import SignalDetection

import unittest
import numpy as np
import matplotlib.pyplot as plt
from sdt.SignalDetection import SignalDetection

class TestSignalDetection(unittest.TestCase):
    def test_d_prime_zero(self):
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=6)

if __name__ == '__main__':
    unittest.main()

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

