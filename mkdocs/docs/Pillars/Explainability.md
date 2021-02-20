# Explainability

## Summary

In many applications, trust in an AI system will come from its ability to ‘explain itself.’ Yet, when it comes to understanding and explaining the inner workings of an algorithm, one size does not fit all. Different stakeholders require explanations for different purposes and objectives, and explanations must be tailored to their needs. A physician might respond best to seeing examples of patient data similar to their patient’s. On the other hand, a developer training a neural net will benefit from seeing how information flows through the algorithm. While a regulator will aim to understand the system as a whole and probe into its logic, consumers affected by a specific decision will be interested only in factors impacting their case – for example, in a loan processing application, they will expect an explanation for why the request was denied and want to understand what changes could lead to approval. IBM Research is creating diverse explanations, including training highly optimized directly interpretable models, creating contrastive explanations of black box models, using information flow in a high-performing complex model to train simpler, interpretable classifiers, learning disentangled representations, and visualizing information flows in neural networks.


## sources 

### Websites 

* IBM explainability: [explainability main page](https://www.research.ibm.com/artificial-intelligence/trusted-ai/#)

* AI Explainability 360: [Toolkit page](http://aix360-dev.mybluemix.net/?_ga=2.110848204.832936263.1613641869-1548554030.1611998814)

* AI Explainability GitHub repo: [AIX360](https://github.com/Trusted-AI/AIX360)

### Papers

* towards Robust Interpretability [paper](https://papers.nips.cc/paper/2018/file/3e9f0fc9b2f89e043bc6233994dfcf76-Paper.pdf)

* Generating Contrastive Explanations withMonotonic Attribute Functions [paper](https://arxiv.org/pdf/1905.12698.pdf)

### Videos

## Taxonomy

Construct a set of measurable metrict which can be used to calculate a score that should indicate how good the explainability of a model is. For The calculation of the score differnt weights can be assigned to the metrics. 


<style>
table {
    width:100%;
}
</style>


| metric     	| description 	| unit             	| weight 	|
|------------	|:-----------:	|------------------	|--------	|
| model type 	|        some models like linear regression or<br> decision tree are very explainable and<br>models like neural networks rather not     	| boolean) 	| 0.6      	|
|faithfulness<br>relevance      	| are the features truly relevant<br>or can some be omitted?       	|           [0,1]       	|    0.2    	|
| monotonicity 	| monotonic attribute functions<br>most important feature for classification           	|           [-1,1]       	|        0.2	|


