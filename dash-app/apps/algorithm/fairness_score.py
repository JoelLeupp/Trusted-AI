# -*- coding: utf-8 -*-
# functions for fairness score

import numpy as np

def score_Statistical_Parity():
    return np.random.randint(1,6)

def score_Disparate_Mistreatment():
    return np.random.randint(1,6)

def score_Class_Imbalance():
    return np.random.randint(1,6)

def score_Biased_Data():
    return np.random.randint(1,6)

def calc_fairness_score():
    
    output = dict(
        Statistical_Parity     = score_Statistical_Parity(),
        Disparate_Mistreatment = score_Disparate_Mistreatment(),
        Disparate_Treatment = score_Disparate_Mistreatment(),
        Disparate_Impact = score_Disparate_Mistreatment(),
        Class_Imbalance        = score_Class_Imbalance(),
        Biased_Data            = score_Biased_Data()
                 )
    
    return output
    
