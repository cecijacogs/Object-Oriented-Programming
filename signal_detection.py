import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
   
    def hit_rate(self):
        # handle empty data
        if self.hits + self.misses == 0:
            return 0.5 
        rate = self.hits / (self.hits + self.misses)
        return min(max(rate, 1e-5), 1 - 1e-5) # avoid extreme values by clipping
    
    def false_alarm(self):
        if self.falseAlarms + self.correctRejections == 0:
            return 0.5
        rate = self.falseAlarms / (self.falseAlarms + self.correctRejections)
        return min(max(rate, 1e-5), 1 - 1e-5) # avoid extreme values by clipping
    
    def d_prime(self):
        # difference between standard deviations of signal and noise distributions as a normal distribution (signal sensitivity)
        # calculate inverse cumulative distribution function of the standard normal distribution.
        # calculate sd of hit rate
        hit_rate_sd = stats.norm.ppf(self.hit_rate())
        # calculate false alarm rate
        false_alarm_sd = stats.norm.ppf(self.false_alarm())
        # calculate d prime
        return hit_rate_sd - false_alarm_sd
        
    def criterion(self): 
        hit_rate_sf = stats.norm.ppf(self.hit_rate())
        false_alarm_sf = stats.norm.ppf(self.false_alarm())
        return -0.5 * (hit_rate_sf + false_alarm_sf)

