# -*- coding: utf-8 -*-
import numpy as np

# functions for methodology score

def score_Normalization():
    return np.random.randint(1,6)

def score_Test_F1_score():
    return np.random.randint(1,6)

def score_Train_Test_Split():
    return np.random.randint(1,6)

def score_Regularization():
    return np.random.randint(1,6)

def score_Test_Accuracy():
    return np.random.randint(1,6)

def calc_methodology_score():
    
    output = dict(
        Normalization    = score_Normalization(),
        Treatment_of_Corrupt_Values     = score_Test_F1_score(),
        Train_Test_Split = score_Train_Test_Split(),
        Regularization   = score_Regularization(),
        Feature_Filtering    = score_Test_Accuracy(),
        Treatment_of_Categorical_Features  = score_Test_Accuracy()
                 )
    return output