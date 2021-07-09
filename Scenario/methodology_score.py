# -*- coding: utf-8 -*-
import numpy as np

def score_Normalization():
    return np.random.randint(1,5)

def score_Test_F1_score():
    return np.random.randint(1,5)

def score_Train_Test_Split():
    return np.random.randint(1,5)

def score_Regularization():
    return np.random.randint(1,5)

def score_Test_Accuracy():
    return np.random.randint(1,5)

def calc_methodology_score():
    
    output = dict(
        Normalization    = score_Normalization(),
        Test_F1_score    = score_Test_F1_score(),
        Train_Test_Split = score_Train_Test_Split(),
        Regularization   = score_Regularization(),
        Test_Accuracy    = score_Test_Accuracy()
                 )
    return output