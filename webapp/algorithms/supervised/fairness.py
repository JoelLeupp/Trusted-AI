# === FAIRNESS ===
import numpy as np
np.random.seed(0)
from helpers import *
from scipy.stats import chisquare
import operator
import funcy

# === Fairness Metrics ===
def analyse(model, training_dataset, test_dataset, factsheet, config): 
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
    
    statistical_parity_difference_thresholds = config["score_statistical_parity_difference"]["thresholds"]["value"]
    overfitting_thresholds = config["score_overfitting"]["thresholds"]["value"]
    underfitting_thresholds = config["score_underfitting"]["thresholds"]["value"]
    equal_opportunity_difference_thresholds = config["score_equal_opportunity_difference"]["thresholds"]["value"]
    average_odds_difference_thresholds = config["score_average_odds_difference"]["thresholds"]["value"]
    disparate_impact_thresholds = config["score_disparate_impact"]["thresholds"]["value"]
    
    output = dict(
        underfitting = underfitting_score(model, training_dataset, test_dataset, factsheet, underfitting_thresholds),
        overfitting = overfitting_score(model, training_dataset, test_dataset, factsheet, overfitting_thresholds),
        statistical_parity_difference = statistical_parity_difference_score(model, training_dataset, factsheet, statistical_parity_difference_thresholds),
        equal_opportunity_difference = equal_opportunity_difference_score(model, test_dataset, factsheet, equal_opportunity_difference_thresholds),
        average_odds_difference = average_odds_difference_score(model, test_dataset, factsheet, average_odds_difference_thresholds),
        disparate_impact = disparate_impact_score(model, test_dataset, factsheet, disparate_impact_thresholds),
        class_balance = class_balance_score(training_dataset, factsheet)
    )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    return  result(score=scores, properties=properties)

# --- Class Balance ---
def class_balance_score(training_data, factsheet):
    """Compute the class balance for the target_column
    If this fails np.nan is returned.

    Args:
        training_data: pd.DataFrame containing the used training data.
        target_column: name of the label column in the provided training data frame.

    Returns:
        On success, returns a score from 1 - 5 for the class balance.
        On failure, a score of np.nan together with an empty properties dictionary is returned.

    """
    try:
        class_balance = class_balance_metric(training_data, factsheet)
        properties = {}
        properties['Metric Description'] = "Measures how well the training data is balanced or unbalanced"
        properties['Depends on'] = 'Training Data'
        if(class_balance == 1):
            score = 5
        else:
            score = 1
        
        properties["Score"] = str(score)
        return result(score=score, properties=properties)     
    except Exception as e:
        print("ERROR in class_balance_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)}) 
    
    
def class_balance_metric(training_data, factsheet):
    """This function runs a statistical test (chisquare) in order to
    check if the targe class labels are equally distributed or not

    Args:
        training_data: pd.DataFrame containing the used training data.
        target_column: name of the label column in the provided training data frame.

    Returns:
        If the label frequency follows a unit distribution a 1 is returned, 0 otherwise.

    """
    try:
        protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)
        absolute_class_occurences = training_data[target_column].value_counts().sort_index().to_numpy()
        significance_level = 0.05
        p_value = chisquare(absolute_class_occurences, ddof=0, axis=0).pvalue

        if p_value < significance_level:
            #The data does not follow a unit distribution
            return 0
        else:
            #We can not reject the null hypothesis assuming that the data follows a unit distribution"
            return 1
    except Exception as e:
        print("ERROR in class_balance_metric(): {}".format(e))
        raise
    
    
# --- Underfitting ---
def underfitting_score(model, training_dataset, test_dataset, factsheet, thresholds):
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
        test_accuracy = compute_accuracy(model, test_dataset, factsheet)
        score = np.digitize(abs(test_accuracy), thresholds, right=False) + 1 

        properties["Test Accuracy"] = "{:.2f}%".format(test_accuracy*100)

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
def overfitting_score(model, training_dataset, test_dataset, factsheet, thresholds):
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
        properties['Depends on'] = 'Model, Training Data, Test Data'
        overfitting_score = np.nan
        training_accuracy = compute_accuracy(model, training_dataset, factsheet)
        test_accuracy = compute_accuracy(model, test_dataset, factsheet)
        # model could be underfitting.
        # for underfitting models the spread is negative
        accuracy_difference = training_accuracy - test_accuracy
        
        underfitting_score = np.digitize(abs(test_accuracy), thresholds, right=False) + 1 
        
        if underfitting_score >= 3:
            overfitting_score = np.digitize(abs(accuracy_difference), thresholds, right=False) + 1 
            properties["Training Accuracy"] = "{:.2f}%".format(training_accuracy*100)
            properties["Test Accuracy"] = "{:.2f}%".format(test_accuracy*100)
            properties["Train Test Accuracy Difference"] = "{:.2f}%".format((training_accuracy - test_accuracy)*100)
        
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
            return result(int(score), properties=properties)
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


# --- Equal Opportunity Difference ---
def equal_opportunity_difference_score(model, test_dataset, factsheet, thresholds):
    """This function computes the equal opportunity difference score.
    It subtracts the true positive rate of the unprotected group from
    the true positive rate of the protected group.
    This is then mapped to a score from one to five using the given thresholds.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.

    Returns:
        A score from one to five representing how fair the calculated
        equal opportunity metric is.
        5 means that the metric is very fair/ low.
        1 means that the metric is very unfair/ high.

    """
    try:
        properties = {}
        score=np.nan
        properties["Metric Description"] = "Difference in true positive rates between protected and unprotected group."
        properties["Depends on"] = "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        tpr_protected, tpr_unprotected, tpr_properties = true_positive_rates(model, test_dataset, factsheet)
        
        properties['----------'] = ''
        properties = properties|tpr_properties 
        equal_opportunity_difference = abs(tpr_protected - tpr_unprotected)
        properties['-----------'] = ''
        
        properties["Formula"] = "Equal Opportunity Difference = |TPR Protected Group - TPR Unprotected Group|"
        properties["Equal Opportunity Difference"] = "{:.2f}%".format(equal_opportunity_difference*100)

        score = np.digitize(abs(equal_opportunity_difference), thresholds, right=False) + 1 
        
        properties["Score"] = str(score)
        return result(score=int(score), properties=properties) 
    except Exception as e:
        print("ERROR in equal_opportunity_difference_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})
 

# --- Average Odds Difference ---
def average_odds_difference_score(model, test_dataset, factsheet, thresholds):
    """This function computes the average odds difference score.
    It subtracts the true positive rate of the unprotected group from
    the true positive rate of the protected group. 
    The same is done for the false positive rates.
    An equally weighted average is formed out of these two differences.
    This is then mapped to a score from one to five using the given thresholds.

    Args:
        model: ML-model.
        training_dataset: pd.DataFrame containing the used training data.
        test_dataset: pd.DataFrame containing the used test data.
        factsheet: json document containing all information about the particular solution.

    Returns:
        A score from one to five representing how fair the calculated
        average odds metric is.
        5 means that the metric is very fair/ low.
        1 means that the metric is very unfair/ high.

    """
    try:
        score = np.nan
        properties = {}
        properties["Metric Description"] = "Is the average of difference in false positive rates and true positive rates between the protected and unprotected group"
        properties["Depends on"] = "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        
        fpr_protected, fpr_unprotected, fpr_properties = false_positive_rates(model, test_dataset, factsheet)
        tpr_protected, tpr_unprotected, tpr_properties = true_positive_rates(model, test_dataset, factsheet)
            
        properties["----------"] = ''
        properties = properties|fpr_properties
        properties = properties|tpr_properties
        properties['-----------'] = ''
        
        average_odds_difference = abs(((tpr_protected - tpr_unprotected) + (fpr_protected - fpr_unprotected))/2)
        properties["Formula"] = "Average Odds Difference = |0.5*(TPR Protected - TPR Unprotected) + 0.5*(FPR Protected - FPR Unprotected)|"
        properties["Average Odds Difference"] = "{:.2f}%".format(average_odds_difference*100)
        
        score = np.digitize(abs(average_odds_difference), thresholds, right=False) + 1 
        
        properties["Score"] = str(score)   
        return result(score=int(score), properties=properties) 
    except Exception as e:
        print("ERROR in average_odds_difference_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})
   
  
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
        
        
def false_positive_rates(model, test_dataset, factsheet):
    """For a given model this function calculates the
    false positive rates for a protected and unprotected group
    based on the given true labels (y_true) and the model predicted
    labels (y_pred)

    Args:
        model: ML-model.
        test_dataset: pd.DataFrame containing the data.
        factsheet: json document containing all information about the particular solution.

    Returns:
        1. false positive prediction rate for observations from the protected group
        2. false positive prediction rate for observations from the unprotected group

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

        #2. Compute the number of negative samples y_true=False for the protected and unprotected group.
        protected_group_true_unfavorable = protected_group[~protected_group[target_column].isin(favorable_outcomes)]
        unprotected_group_true_unfavorable = unprotected_group[~unprotected_group[target_column].isin(favorable_outcomes)]
        protected_group_n_true_unfavorable = len(protected_group_true_unfavorable)
        unprotected_group_n_true_unfavorable = len(unprotected_group_true_unfavorable)

        #3. Calculate the number of false positives for the protected and unprotected group
        protected_group_true_unfavorable_pred_favorable = protected_group_true_unfavorable[protected_group_true_unfavorable['y_pred'].isin(favorable_outcomes)]
        unprotected_group_true_unfavorable_pred_favorable = unprotected_group_true_unfavorable[unprotected_group_true_unfavorable['y_pred'].isin(favorable_outcomes)]
        protected_group_n_true_unfavorable_pred_favorable = len(protected_group_true_unfavorable_pred_favorable)
        unprotected_group_n_true_unfavorable_pred_favorable = len(unprotected_group_true_unfavorable_pred_favorable)

        #4. Calculate fpr for both groups.
        fpr_protected = protected_group_n_true_unfavorable_pred_favorable/protected_group_n_true_unfavorable
        fpr_unprotected = unprotected_group_n_true_unfavorable_pred_favorable/unprotected_group_n_true_unfavorable
        
        #5. Adding properties
        properties["|{x|x is protected, y_true is unfavorable, y_pred is favorable}|"] = protected_group_n_true_unfavorable_pred_favorable
        properties["|{x|x is protected, y_true is Unfavorable}|"] = protected_group_n_true_unfavorable
        properties["FPR Protected Group"] = "P(y_pred is favorable|y_true is unfavorable, protected=True) = {:.2f}%".format(fpr_protected*100) 
        properties["|{x|x is not protected, y_true is unfavorable, y_pred is favorable}|"] = unprotected_group_n_true_unfavorable_pred_favorable
        properties["|{x|x is not protected, y_true is unfavorable}|"] = unprotected_group_n_true_unfavorable
        properties["FPR Unprotected Group"] = "P(y_pred is favorable|y_true is unfavorable, protected=False) = {:.2f}%".format(fpr_unprotected*100)
            
        return fpr_protected, fpr_unprotected, properties

    except Exception as e:
        print("ERROR in false_positive_rates(): {}".format(e))
        raise
   

def true_positive_rates(model, test_dataset, factsheet):
    """For a given model this function calculates the
    true positive rates for a protected and unprotected group
    based on the given true labels (y_true) and the model predicted
    labels (y_pred)

    Args:
        model: ML-model.
        test_dataset: pd.DataFrame containing the data.
        factsheet: json document containing all information about the particular solution.

    Returns:
        1. true positive prediction rate for observations from the protected group
        2. true positive prediction rate for observations from the unprotected group

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

        favored_samples = data[data[target_column].isin(favorable_outcomes)]
        protected_favored_samples = favored_samples[favored_samples[protected_feature].isin(protected_values)]
        unprotected_favored_samples = favored_samples[~favored_samples[protected_feature].isin(protected_values)]

        num_unprotected_favored_true = len(unprotected_favored_samples)
        num_unprotected_favored_pred = len(unprotected_favored_samples[unprotected_favored_samples['y_pred'].isin(favorable_outcomes)])
        tpr_unprotected = num_unprotected_favored_pred/num_unprotected_favored_true

        num_protected_favored_true = len(protected_favored_samples)
        num_protected_favored_pred = len(protected_favored_samples[protected_favored_samples['y_pred'].isin(favorable_outcomes)])
        tpr_protected = num_protected_favored_pred / num_protected_favored_true 
        
        # Adding properties
        properties["|{x|x is protected, y_true is favorable, y_pred is favorable}|"] = num_protected_favored_pred
        properties["|{x|x is protected, y_true is favorable}|"] = num_protected_favored_true
        properties["TPR Protected Group"] = "P(y_pred is favorable|y_true is favorable, protected=True) = {:.2f}%".format(tpr_protected*100) 
        properties["|{x|x is not protected, y_true is favorable, y_pred is favorable}|"] = num_unprotected_favored_pred
        properties["|{x|x is not protected, y_true is favorable}|"] = num_unprotected_favored_true
        properties["TPR Unprotected Group"] = "P(y_pred is favorable|y_true is favorable, protected=False) = {:.2f}%".format(tpr_unprotected*100)
        
        return tpr_protected, tpr_unprotected, properties

    except Exception as e:
        print("ERROR in true_positive_rates(): {}".format(e))
        raise



    
