# -*- coding: utf-8 -*-
import numpy as np

# functions for robustness score

def score_Confidence_Score():
    return np.random.randint(1,6)

def score_Class_Specific_Metrics():
    return np.random.randint(1,6)

def calc_robustness_score():
    
    output = dict(
        Confidence_Score          = score_Confidence_Score(),
        Clique_Method    = score_Class_Specific_Metrics(),
        Loss_Sensitivity   = score_Class_Specific_Metrics(),
        CLEVER_Score   = score_Class_Specific_Metrics()
                 )
    return output
