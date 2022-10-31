# === Explainability ===
import numpy as np
import pandas as pd
import json
import collections

result = collections.namedtuple('result', 'score properties')
info = collections.namedtuple('info', 'description value')


def analyse(clf, train_data, test_data, config, factsheet):
    
    #function parameters
    target_column = factsheet["general"].get("target_column")
    clf_type_score = config["score_algorithm_class"]["clf_type_score"]["value"]
    ms_thresholds = config["score_model_size"]["thresholds"]["value"]
    cf_thresholds = config["score_correlated_features"]["thresholds"]["value"]
    high_cor = config["score_correlated_features"]["high_cor"]["value"]
    fr_thresholds = config["score_feature_relevance"]["thresholds"]["value"]
    threshold_outlier = config["score_feature_relevance"]["threshold_outlier"]["value"]
    penalty_outlier = config["score_feature_relevance"]["penalty_outlier"]["value"]
    
    output = dict(
        algorithm_class     = algorithm_class_score(clf, clf_type_score),
        correlated_features = correlated_features_score(train_data, test_data, thresholds=cf_thresholds, target_column=target_column, high_cor=high_cor ),
        model_size          = model_size_score(train_data, ms_thresholds),
        feature_relevance   = feature_relevance_score(clf, train_data ,target_column=target_column, thresholds=fr_thresholds,
                                                     threshold_outlier =threshold_outlier,penalty_outlier=penalty_outlier )
                 )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)


def algorithm_class_score(clf, clf_type_score):

    clf_name = type(clf).__name__
    exp_score = clf_type_score.get(clf_name,np.nan)
    properties= {"dep" :info('Depends on','Model'),
        "clf_name": info("model type",clf_name)}
    
    return  result(score=exp_score, properties=properties)

def correlated_features_score(train_data, test_data, thresholds=[0.05, 0.16, 0.28, 0.4], target_column=None, high_cor=0.9):
    
    test_data = test_data.copy()
    train_data = train_data.copy()
     
    if target_column:
        X_test = test_data.drop(target_column, axis=1)
        X_train = train_data.drop(target_column, axis=1)
    else:
        X_test = test_data.iloc[:,:-1]
        X_train = train_data.iloc[:,:-1]
        
    
    df_comb = pd.concat([X_test, X_train])
    corr_matrix = df_comb.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > high_cor)]
    
    pct_drop = len(to_drop)/len(df_comb.columns)
    
    bins = thresholds
    score = 5-np.digitize(pct_drop, bins, right=True) 
    properties= {"dep" :info('Depends on','Training Data'),
        "pct_drop" : info("Percentage of highly correlated features", "{:.2f}%".format(100*pct_drop))}
    
    return  result(score=int(score), properties=properties)


def model_size_score(test_data, thresholds = np.array([10,30,100,500])):
    
    dist_score = 5- np.digitize(test_data.shape[1]-1 , thresholds, right=True) 
         
    return result(score=int(dist_score), properties={"dep" :info('Depends on','Training Data'),
        "n_features": info("number of features", test_data.shape[1]-1)})

def feature_relevance_score(clf, train_data, target_column=None, threshold_outlier = 0.03, penalty_outlier = 0.5, thresholds = [0.05, 0.1, 0.2, 0.3]):
    
    pd.options.mode.chained_assignment = None  
    train_data = train_data.copy()
    if target_column:
        X_train = train_data.drop(target_column, axis=1)
        y_train = train_data[target_column]
    else:
        X_train = train_data.iloc[:,:-1]
        y_train = train_data.iloc[:,-1: ]
        
    scale_factor = 1.5
    distri_threshold = 0.6
    if (type(clf).__name__ == 'LogisticRegression') or (type(clf).__name__ == 'LinearRegression'): 
        #normalize 
        #for feature in X_train.columns:
        #    X_train.loc[feature] = X_train[feature] / X_train[feature].std()
        clf.max_iter =1000
        clf.fit(X_train, y_train.values.ravel())
        importance = clf.coef_[0]
        #pd.DataFrame(columns=feat_labels,data=importance.reshape(1,len(importance))).plot.bar()
        
    elif  (type(clf).__name__ == 'RandomForestClassifier') or (type(clf).__name__ == 'DecisionTreeClassifier'):
         importance=clf.feature_importances_
         
    else:
        return result(score= np.nan, properties={"dep" :info('Depends on','Training Data and Model')}) 
   
    # absolut values
    importance = abs(importance)
    
    feat_labels = X_train.columns
    indices = np.argsort(importance)[::-1]
    feat_labels = feat_labels[indices]

    importance = importance[indices]
    
    # calculate quantiles for outlier detection
    q1, q2, q3 = np.percentile(importance, [25, 50 ,75])
    lower_threshold , upper_threshold = q1 - scale_factor*(q3-q1),  q3 + scale_factor*(q3-q1) 
    
    #get the number of outliers defined by the two thresholds
    n_outliers = sum(map(lambda x: (x < lower_threshold) or (x > upper_threshold), importance))
    
    # percentage of features that concentrate distri_threshold percent of all importance
    pct_dist = sum(np.cumsum(importance) < 0.6) / len(importance)
    
    dist_score = np.digitize(pct_dist, thresholds, right=False) + 1 
    
    if n_outliers/len(importance) >= threshold_outlier:
        dist_score -= penalty_outlier
    
    score =  max(dist_score,1)
    properties = {"dep" :info('Depends on','Training Data and Model'),
        "n_outliers":  info("number of outliers in the importance distribution",int(n_outliers)),
                  "pct_dist":  info("percentage of feature that make up over 60% of all features importance", "{:.2f}%".format(100*pct_dist)),
                  "importance":  info("feature importance", {"value": list(importance), "labels": list(feat_labels)})
                  }
    
    return result(score=int(score), properties=properties)
    # import seaborn as sns
    # sns.boxplot(data=importance)
    
