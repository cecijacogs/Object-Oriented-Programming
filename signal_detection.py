import numpy as np
import scipy.stats as stats

class SignalDetection(hits, misses, falseAlarms, correctRejections):
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
   
    def hit_rate(self):
        return self.hits / (self.hits + self.misses)
    
    def false_alarm(falseAlarms, correctRejections):
        return falseAlarms / (falseAlarms + correctRejections)
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

