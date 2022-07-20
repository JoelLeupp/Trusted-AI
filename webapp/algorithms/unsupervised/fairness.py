# === FAIRNESS ===
import numpy as np
import keras
from sklearn.ensemble import IsolationForest
from keras.models import Sequential


np.random.seed(0)
from helpers import *
from scipy.stats import chisquare
import operator
import funcy

# === Fairness Metrics ===
def analyse(model, training_dataset, test_dataset, outliers_dataset, factsheet, config):
    """Triggers the fairness analysis and in a first step all fairness metrics get computed.
    In a second step, the scores for the fairness metrics are then created from
    mapping every metric value to a respective score.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.
        config: Config file containing the threshold values for the metrics.

    Returns:
        Returns a result object containing all metric scores
        and matching properties for every metric

    """
    print_details = True
    outlier_percentage = 0.1

    if isKerasAutoencoder(model):
        print("train size: ", training_dataset.shape)
        print("test size: ", test_dataset.shape)
        print("outliers size: ", outliers_dataset.shape)
        print("model size: ", model.summary())

        outlier_thresh = get_threshold_mse_iqr(model, training_dataset)
    else:
        outlier_thresh = 0

    statistical_parity_difference_thresholds = config["score_statistical_parity_difference"]["thresholds"]["value"]
    overfitting_thresholds = config["score_overfitting"]["thresholds"]["value"]
    underfitting_thresholds = config["score_underfitting"]["thresholds"]["value"]
    disparate_impact_thresholds = config["score_disparate_impact"]["thresholds"]["value"]

    output = dict(
        underfitting = underfitting_score(model, training_dataset, test_dataset, factsheet, underfitting_thresholds, outlier_thresh, print_details),
        overfitting = overfitting_score(model, training_dataset, test_dataset, outliers_dataset, factsheet, outlier_percentage, overfitting_thresholds, outlier_thresh, print_details),
        #statistical_parity_difference = statistical_parity_difference_score(model, training_dataset, factsheet, statistical_parity_difference_thresholds, print_details),
        #disparate_impact=disparate_impact_score(model, test_dataset, factsheet, disparate_impact_thresholds, print_details),

        statistical_parity_difference = result(score=int(1), properties={}),
        disparate_impact = result(score=int(1), properties={}),
    )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return result(score=scores, properties=properties)
    
# --- Underfitting ---
def underfitting_score(model, training_dataset, test_dataset, factsheet, thresholds, outlier_thresh, print_details = False):
    """This function computes the training and test accuracy for the given model and then
    compares how much lower the training accuracy is than the test accuracy.
    If this is the case then we consider our model to be underfitting.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.
        thresholds: Threshold values used to determine the final score

    Returns:
        A score from one to five representing the level of underfitting.
        5 means the model is not underfitting
        1 means the model is strongly underfitting

    """
    try:
        properties = {}
        properties['Metric Description'] = "Compares the models achieved test accuracy against a baseline."
        properties['Depends on'] = 'Model, Test Data'
        score = 0

        detection_ratio_train = compute_outlier_ratio(model, training_dataset, outlier_thresh)
        detection_ratio_val = compute_outlier_ratio(model, test_dataset, outlier_thresh)
        #detection_ratio_val = compute_outlier_ratio(model, val_data, outlier_thresh)

        perc_diff = abs(detection_ratio_train - detection_ratio_val)
        score = np.digitize(perc_diff, thresholds, right=False) + 1

        if print_details:
            print("\t   UNDERFITTING DETAILS")
            print("\t model is AutoEncoder: ", isKerasAutoencoder(model))
            print("\t model is IsolationForest: ", isIsolationForest(model))
            print("\t detected outlier ratio in training data: %.4f" % detection_ratio_train)
            print("\t detected outlier ratio in validation data: %.4f" % detection_ratio_val)
            print("\t absolute difference: %.4f" % perc_diff)

        properties["Test Accuracy"] = "{:.2f}%".format(detection_ratio_val*100)

        if score == 5:
            properties["Conclusion"] = "Model is not underfitting"
        elif score == 4:
            properties["Conclusion"] = "Model mildly underfitting"
        elif score == 3:
            properties["Conclusion"] = "Model is slighly underfitting"
        elif score == 2:
            properties["Conclusion"] = "Model is underfitting"
        else:
            properties["Conclusion"] = "Model is strongly underfitting"

        properties["Score"] = str(score)
        return result(score=int(score), properties=properties)
    
    except Exception as e:
        print("ERROR in underfitting_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)}) 

    
# --- Overfitting ---
def overfitting_score(model, training_dataset, test_dataset, outliers_dataset, factsheet, outlier_percentage, thresholds, outlier_thresh, print_details = False):
    """This function computes the training and test accuracy for the given model and then
    compares how much higher the training accuracy is compared to the test accuracy.
    If this is the case then we consider our model to be overfitting.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.
        thresholds: Threshold values used to determine the final score

    Returns:
        A score from one to five representing the level of overfitting.
        5 means the model is not overfitting
        1 means the model is strongly overfitting

    """

    try:
        properties = {}
        properties['Metric Description'] = "Overfitting is present if the training accuracy is significantly higher than the test accuracy"
        properties['Depends on'] = 'Model, Training Data, Test Data, Outliers Data'

        #compute underfitting score
        detection_ratio_train = compute_outlier_ratio(model, training_dataset, outlier_thresh)
        detection_ratio_test = compute_outlier_ratio(model, test_dataset, outlier_thresh)

        perc_diff = abs(detection_ratio_train - detection_ratio_test)
        underfitting_score = np.digitize(perc_diff, [0.1,0.05,0.025,0.01], right=False) + 1

        #if underfitting_score >= 3:
        if 3 >= 3:
            # compute outlier ratio in outlier dataset
            detection_ratio_outliers = compute_outlier_ratio(model, outliers_dataset, outlier_thresh)

            # compute outlier ratio in train dataset
            detection_ratio_test = compute_outlier_ratio(model, test_dataset, outlier_thresh)

            perc_diff = abs(outlier_percentage - detection_ratio_test)
            training_accuracy = abs(outlier_percentage - perc_diff) / outlier_percentage

            mean = (detection_ratio_outliers + training_accuracy) / 2
            overfitting_score = np.digitize(mean, thresholds, right=False) + 1

            properties["Outliers Accuracy"] = "{:.2f}%".format(detection_ratio_outliers*100)
            properties["Test Accuracy"] = "{:.2f}%".format(detection_ratio_test*100)
            properties["Outliers Test Accuracy Difference"] = "{:.2f}%".format(perc_diff*100)

            if print_details:
                print("\t   OVERFITTING DETAILS")
                print("\t   outlier percentage in training data: ", outlier_percentage)
                print("\t   detected outlier ratio in validation dataset: %.4f" % detection_ratio_test)
                print("\t   training accuracy: %.4f" % training_accuracy)
                print("\t   detected outlier ratio in outlier dataset: %.4f" % detection_ratio_outliers)
                print("\t   mean value: %.4f" % mean)

            if overfitting_score == 5:
                properties["Conclusion"] = "Model is not overfitting"
            elif overfitting_score == 4:
                properties["Conclusion"] = "Model mildly overfitting"
            elif overfitting_score == 3:
                properties["Conclusion"] = "Model is slighly overfitting"
            elif overfitting_score == 2:
                properties["Conclusion"] = "Model is overfitting"
            else:
                properties["Conclusion"] = "Model is strongly overfitting"

            properties["Score"] = str(overfitting_score)
            return result(int(overfitting_score), properties=properties)
        else:
            return result(overfitting_score, properties={"Non computable because": "The test accuracy is to low and if the model is underfitting to much it can't be overfitting at the same time."})
    except Exception as e:
        print("ERROR in overfitting_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)}) 


# --- Statistical Parity Difference ---
def statistical_parity_difference_score(model, training_dataset, factsheet, thresholds):
    """This function scores the computed statistical parity difference
    on a scale from 1 (very bad) to 5 (very good)

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        factsheet: json document containing all information about the particular solution.
        thresholds: Threshold values used to determine the final score

    Returns:
        A score from one to five representing how fair the calculated
        statistical parity differnce is.
        5 means that the statistical parity difference is very fair/ low.
        1 means that the statistical parity difference is very unfair/ high.

    """
    try: 
        
        score = np.nan
        properties = {}
        properties["Metric Description"] = "The spread between the percentage of observations from the majority group receiving a favorable outcome compared to the protected group. The closes this spread is to zero the better."
        properties["Depends on"] = "Training Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        statistical_parity_difference, spdm_properties = statistical_parity_difference_metric(model, training_dataset, factsheet)

        properties['----------'] = ''
        properties = properties|spdm_properties
        properties['-----------'] = ''
        properties["Formula"] =  "Statistical Parity Difference = |Favored Protected Group Ratio - Favored Unprotected Group Ratio|"
        properties["Statistical Parity Difference"] = "{:.2f}%".format(statistical_parity_difference*100)
        
        score = np.digitize(abs(statistical_parity_difference), thresholds, right=False) + 1 
        
        properties["Score"] = str(score)
        return result(score=int(score), properties=properties)
    except Exception as e:
        print("ERROR in statistical_parity_difference_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})

    
def statistical_parity_difference_metric(model, training_dataset, factsheet):
    """This function computes the statistical parity metric.
    It separates the data into a protected and an unprotected group 
    based on the given definition for protected groups.
    Then it computes the ratio of individuals receiving a favorable outcome
    divided by the total number of observations in the group.
    This is done both for the protected and unprotected group.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        factsheet: json document containing all information about the particular solution.

    Returns:
        1. The ratio of people from the unprotected group receiving a favorable outcome.
        2. The ratio of people from the protected group receiving a favorable outcome.
        3. The statistical parity differenc as the difference of the previous two values.

    """
    try: 
        properties = {}
        protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)
        
        minority = training_dataset[training_dataset[protected_feature].isin(protected_values)]
        minority_size = len(minority)
        majority = training_dataset[~training_dataset[protected_feature].isin(protected_values)]
        majority_size = len(majority)

        favored_minority = minority[minority[target_column].isin(favorable_outcomes)]
        favored_minority_size = len(favored_minority)

        favored_minority_ratio = favored_minority_size/minority_size

        favored_majority = majority[majority[target_column].isin(favorable_outcomes)]
        favored_majority_size = len(favored_majority)
        favored_majority_ratio = favored_majority_size/majority_size
        
        properties["|{x|x is protected, y_true is favorable}|"] = favored_minority_size
        properties["|{x|x is protected}|"] = minority_size
        properties["Favored Protected Group Ratio"] =  "P(y_true is favorable|protected=True) = {:.2f}%".format(favored_minority_ratio*100)
        properties["|{x|x is not protected, y_true is favorable}|"] = favored_majority_size
        properties["|{x|x is not protected}|"] = majority_size
        properties["Favored Unprotected Group Ratio"] =  "P(y_true is favorable|protected=False) = {:.2f}%".format(favored_majority_ratio*100)
        
        statistical_parity_difference = abs(favored_minority_ratio - favored_majority_ratio)
        return statistical_parity_difference, properties
    except Exception as e:
        print("ERROR in statistical_parity_difference_metric(): {}".format(e))
        raise
  
# --- Disparate Impact ---
def disparate_impact_score(model, test_dataset, factsheet, thresholds):
    """This function computes the disparate impact score.
    It divides the ratio of favored people within the protected group by
    the ratio of favored people within the unprotected group in order
    to receive the disparate impact ratio.
    This is then mapped to a score from one to five using the given thresholds.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.

    Returns:
        A score from one to five representing how fair the calculated
        disparate impact metric is.
        5 means that the disparate impact metric is very fair/ low.
        1 means that the disparate impact metric is very unfair/ high.

    """
    try:
        score = np.nan
        properties = {}
        properties["Metric Description"] = "Is quotient of the ratio of samples from the protected group receiving a favorable prediction divided by the ratio of samples from the unprotected group receiving a favorable prediction"
        properties["Depends on"] = "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        disparate_impact, dim_properties = disparate_impact_metric(model, test_dataset, factsheet)
        
        properties["----------"] = ''
        properties = properties|dim_properties
        properties['-----------'] = ''
        
        properties["Formula"] = "Disparate Impact = Protected Favored Ratio / Unprotected Favored Ratio"
        properties["Disparate Impact"] = "{:.2f}".format(disparate_impact)

        score = np.digitize(disparate_impact, thresholds, right=False)+1
            
        properties["Score"] = str(score)
        return result(score=int(score), properties=properties) 
    except Exception as e:
        print("ERROR in disparate_impact_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})


def disparate_impact_metric(model, test_dataset, factsheet):
    """This function computes the disparate impact metric.
    It separates the data into a protected and an unprotected group 
    based on the given definition for protected groups.
    Then it computes the ratio of individuals receiving a favorable outcome
    divided by the total number of observations in the group.
    This is done both for the protected and unprotected group.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.

    Returns:
        1. The ratio of people from the protected group receiving a favorable outcome.
        2. The ratio of people from the unprotected group receiving a favorable outcome.

    """
    try: 
        properties = {}
        data = test_dataset.copy(deep=True)
        
        protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)
        
        X_data = data.drop(target_column, axis=1)
        if (isinstance(model, tf.keras.Sequential)):
            y_pred_proba = model.predict(X_data)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_data).flatten()
        data['y_pred'] = y_pred.tolist()

        protected_group = data[data[protected_feature].isin(protected_values)]
        unprotected_group = data[~data[protected_feature].isin(protected_values)]
        protected_group_size = len(protected_group)
        unprotected_group_size = len(unprotected_group)

        protected_favored_group = protected_group[protected_group['y_pred'].isin(favorable_outcomes)]
        unprotected_favored_group = unprotected_group[unprotected_group['y_pred'].isin(favorable_outcomes)]
        protected_favored_group_size = len(protected_favored_group)
        unprotected_favored_group_size = len(unprotected_favored_group)
        
        protected_favored_ratio = protected_favored_group_size / protected_group_size
        unprotected_favored_ratio = unprotected_favored_group_size / unprotected_group_size
        
        properties["|{x|x is protected, y_pred is favorable}"] = protected_favored_group_size
        properties["|{x|x is protected}|"] = protected_group_size
        properties["Protected Favored Ratio"] = "P(y_hat=favorable|protected=True) = {:.2f}%".format(protected_favored_ratio*100)
        properties["|{x|x is not protected, y_pred is favorable}|"] = unprotected_favored_group_size
        properties["|{x|x is not protected}|"] = unprotected_group_size
        properties["Unprotected Favored Ratio"] = "P(y_hat=favorable|protected=False) = {:.2f}%".format(unprotected_favored_ratio*100) 

        disparate_impact = abs(protected_favored_ratio / unprotected_favored_ratio)
        return disparate_impact, properties

    except Exception as e:
        print("ERROR in disparate_impact_metric(): {}".format(e))
        raise

        
# --- Helper Functions ---
'''
def compute_accuracy(model, dataset, factsheet):
    """This function computes and returns 
    the accuracy a model achieves on a given dataset

    Args:
        model: ML-model.
        dataset: pd.DataFrame containing the data.
        factsheet: json document containing all information about the particular solution.

    Returns:
        The prediction accuracy that the model achieved on the given data.

    """
    try:
        protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)
        X_data = dataset.drop(target_column, axis=1)
        y_data = dataset[target_column]

        y_true = y_data.values.flatten()
        if (isinstance(model, tf.keras.Sequential)):
            y_train_pred_proba = model.predict(X_train)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_data).flatten()
        return metrics.accuracy_score(y_true, y_pred)
    except Exception as e:
        print("ERROR in compute_accuracy(): {}".format(e))
        raise
'''


def compute_accuracy(unique_elements, counts_elements, outlier_indicator=False, normal_indicator=True):
    tot_datapoints = 0
    num_outliers = 0
    num_normal = 0

    for i, el in enumerate(unique_elements):
        if el == normal_indicator:
            num_normal = counts_elements.item(i)
            tot_datapoints += num_normal
        if el == outlier_indicator:
            num_outliers = counts_elements.item(i)
            tot_datapoints += num_outliers

    if (tot_datapoints > 0):
        accuracy = num_outliers / tot_datapoints

    else:
        accuracy = 0

    return accuracy


def isKerasAutoencoder(model):
    return isinstance(model, keras.engine.functional.Functional)


def isIsolationForest(model):
    return isinstance(model, IsolationForest)


def compute_outlier_ratio(model, data, outlier_thresh, print_details=False):
    if isKerasAutoencoder(model):
        mad_outliers = detect_outliers(model, data, outlier_thresh)
        unique_elements, counts_elements = np.unique(mad_outliers, return_counts=True)
        outlier_detection_percentage = compute_accuracy(unique_elements, counts_elements)

    elif isIsolationForest(model):
        mad_outliers = model.predict(data)
        unique_elements, counts_elements = np.unique(mad_outliers, return_counts=True)
        outlier_detection_percentage = compute_accuracy(unique_elements, counts_elements, -1, 1)

    else:
        mad_outliers = model.predict(data)
        unique_elements, counts_elements = np.unique(mad_outliers, return_counts=True)
        outlier_detection_percentage = compute_accuracy(unique_elements, counts_elements, 1, 0)

    if print_details:
        print("\t uniqueelements: ", unique_elements)
        print("\t counts elements: ", counts_elements)

    return outlier_detection_percentage

#Predict outliers in "df" using "autoencoder" model and "threshold_mse" as anomaly limit
def detect_outliers(autoencoder, df, threshold_mse):
    if(len(threshold_mse)==2):
        return detect_outliers_range(autoencoder, df, threshold_mse)
    pred=autoencoder.predict(df)
    mse = np.mean(np.power(df - pred, 2), axis=1)
    outliers = [np.array(mse) < threshold_mse]
    return outliers

def detect_outliers_range(autoencoder, df, threshold_mse):
    pred=autoencoder.predict(df)
    mse = np.mean(np.power(df - pred, 2), axis=1)
    up_bound = threshold_mse[0]
    bottom_bound = threshold_mse[1]
    outliers = [(np.array(mse) < up_bound)&(np.array(mse) > bottom_bound)]
    return outliers
    
#Get anomaly threshold from "autoencoder" setting the threshold in Q1,Q3+-1.5IQR
def get_threshold_mse_iqr(autoencoder,train_data):
    train_predicted = autoencoder.predict(train_data)
    mse = np.mean(np.power(train_data - train_predicted, 2), axis=1)
    iqr = np.quantile(mse,0.75) - np.quantile(mse, 0.25) # interquartile range
    up_bound = np.quantile(mse,0.75) + 1.5*iqr
    bottom_bound = np.quantile(mse,0.25) - 1.5*iqr
    thres = [up_bound,bottom_bound]
    return thres
