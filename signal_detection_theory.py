import unittest 
import numpy as np 
import scipy
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
        false_alarm_rate = min(max(hit_rate, 1e-10), 1 - 1e-10)

        return norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
    
    def criterion(self):
        hit_rate = self.hit_rate()
        false_alarm_rate = self.false_alarm_rate()

        hit_rate = min(max(hit_rate, 1e-10), 1 - 1e-10)
        false_alarm_rate = min(max(false_alarm_rate, 1e-10), 1 - 1e-10)

        return -0.5 * (norm.ppf(hit_rate) + norm.ppf(false_alarm_rate))

