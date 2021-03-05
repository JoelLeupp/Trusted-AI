# Explainability

## Introduction (IBM)

In many applications, trust in an AI system will come from its ability to ‘explain itself.’ Yet, when it comes to understanding and explaining the inner workings of an algorithm, one size does not fit all.Different stakeholders require explanations for different purposes and objectives, and explanations must be tailored to their needs. A physician might respond best to seeing examples of patient data similar to their patient’s. On the other hand, a developer training a neural net will benefit from seeing how information flows through the algorithm. While a regulator will aim to understand the system as a whole and probe into its logic, consumers affected by a specific decision will be interested only in factors impacting their case – for example, in a loan processing application, they will expect an explanation for why the request was denied and want to understand what changes could lead to approval. IBM Research is creating diverse explanations, including training highly optimized directly interpretable models, creating contrastive explanations of black box models, using information flow in a high-performing complex model to train simpler, interpretable classifiers, learning disentangled representations, and visualizing information flows in neural networks.


## Summaries of Paper & Website

### [Explainable AI](https://www.pwc.co.uk/audit-assurance/assets/explainable-ai.pdf)

The paper is business oriented and talks about why explainability is an advatage to have embedded in AI
algorithms and in which usecases explainability should have a high priority. 

* (p.8) The need for Explainability: Working as intended?, How sensitive is the impact?, Are you comfortable with
the level of control?

* (p.9) factors to consider: Revenue, Rate, Rigour, Regulation, Reputation and Risk

* (p.12) Explainable by design: Explainability needs to be considered up front and embedded into the design of
the AI application. It affects the choice ofmachine learning algorithm and mayimpact the way you choose to pre-process data.

* (p.13) Trade-offs in explainability: Interpretability is a characteristic of a model that is generally considered to come at a cost. As a rule of thumb, the more complex the model, the more accurate it is, but the less interpretable it is. (Decision tree vs. DNN with many layers and features)

* (p.14) Differnt explaination techniques: Sensitivity analysis,Local Interpretable Model Explanations (LIME), Shapley Additive Explanations (SHAP), Tree interpreters, Neural Network Interpreters

* (Appendix 2)  Subjective scale of explainability of different classes of algorithms and learning techniques 
(with 1 being the most difficult and 5 being the easiest to explain) see a compriesed version of the table belwo:

| Learning technique              | Scale of explainability (1-5) |
|---------------------------------|:-----------------------------:|
| Bayesian belief networks (BNNs) |              3.5              |
| Decision trees                  |               4               |
| Logistic regression             |               3               |
| Support vector machines (SVMs)  |               2               |
| K-means clustering              |               3               |
| Neural networks                 |               1               |
| Random forest/boosting          |               3               |
| Q-learning                      |               2               |
| Hidden Markov models            |               3               |

<br>

### [Explainable Artificial Intelligence (XAI)](https://www.darpa.mil/program/explainable-artificial-intelligence) 

XAI is a project from Defense Advanced Research Projects Agency which aims to produce more explainable models, while maintaining a high level of learning performance (prediction accuracy)and enable human users to understand, appropriately trust, and effectively manage the emerging generation of artificially intelligent partners

see their concept here: [DARPA slide deck on explainability](https://www.darpa.mil/attachments/XAIIndustryDay_Final.pptx) (two key slides below)


![DARPA Concept](images/DARPA_concept.png)

![DARPA Modles](images/DARPA_explaibale_models.png)



## Taxonomy

Construct a set of measurable metrict which can be used to calculate a score that should indicate how good the explainability of a model is. For The calculation of the score differnt weights can be assigned to the metrics. 


<style>
table {
    width:100%;
}
</style>


| metric     	| description 	| unit             	| weight 	|
|------------	|:-----------:	|------------------	|--------	|
| model type 	|        some models like linear regression or<br> decision tree are very explainable and<br>models like neural networks less    	| [1,5] 	| 0.6      	|
|faithfulness<br>relevance      	| are the features truly relevant<br>or can some be omitted?       	|           [0,1]       	|    0.2    	|
| monotonicity 	| monotonic attribute functions<br>most important feature for classification           	|           [-1,1]       	|        0.2	|


