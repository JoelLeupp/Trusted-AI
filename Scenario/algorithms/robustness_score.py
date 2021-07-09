# -*- coding: utf-8 -*-
import numpy as np

def score_Confidence_Score():
    return np.random.randint(1,5)

def score_Class_Specific_Metrics():
    return np.random.randint(1,5)

def calc_robustness_score():
    
    output = dict(
        Confidence_Score          = score_Confidence_Score(),
        Class_Specific_Metrics   = score_Class_Specific_Metrics()
                 )
    return output
