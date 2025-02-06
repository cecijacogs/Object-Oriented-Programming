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
        
