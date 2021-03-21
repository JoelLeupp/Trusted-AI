# Robustness

## Introduction
Machine Learning models might not be robust against adversarial perturbations. With visually imperceptible adversarial perturbations, attackers may cause an input to be misclassified. These attacks can be generated in both the black-box setting, where the parameters of the model are unknown to the attacker, and the white-box settings, where attacker have the all necessary information about the model. To trust AI we need to be sure that it is robust against adversarial attacks.

## Key Takeaways, Questions and Limitations
* *Key Takeaways*
    * Robustness is a mathematical term and context independent.
    * Mostly measured by a specific adversarial attack's success rate on the model.
* *Limitations*
    * Robustness can be defined as vulnerability to adversarial examples. An adversarial example is an instance with small, intentional feature perturbations that cause a machine learning model to make a false prediction. Hence robustness is defined on models where the "False Prediction" is clearly defined and where true and false predictions are sufficiently different. For example we can talk about robustness of classifiers. However it is quite difficult to talk about robustness of regression models.  
    *  Robustness only makes sense if the input to the model is sufficiently large. When the input is adversarially perturbed the difference should be unnoticiable to the human eye but still be large enough to make the input misclassified. This is only possible when the input is high dimensional so that the differences can be hidden. Thats why they usually use Convolutional Neural Networks as an example.
    * The topic is well explored on neural network classifiers. For the other models I was able to find a paper proposing an attack working for neural networks, logistic regression models and SVM's. With the help of the same paper, I learned that 2 other attacks can be modified to work for logistic regression, SVM and neural networks. We should narrow down the possible different classifier models that can be used with our algorithm.


## Sources
### Websites 

* IBM Robustness: [Robustness main page](https://www.research.ibm.com/artificial-intelligence/publications/?researcharea=robustness)

* Adversarial Robustness 360: [Toolbox](http://art360.mybluemix.net/)

* Adversarial Robustness Toolbox Github repo: [ART v1.5](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

* CLEVER Github repo: [Robustness Score](https://github.com/IBM/CLEVER-Robustness-Score)


### Papers and Summaries
| Year | Author | Title | Content |
|------|--------|-------|---------|
| 2019 |Maria-Irina Nicolae et al.|[Adversarial Robustness Toolbox v1.0.0](https://arxiv.org/pdf/1807.01069.pdf)| Adversarial Robustness Toolbox is an open-source Python library. It offers adversarial attack/defense implementations, runtime detection methods, poisoning detection and robustness metrics.|
| 2018| Tsui-Wei Weng et al.| [Evaluating The Robustness of Neural Networks: An Extreme Value Theory Approach](https://openreview.net/pdf?id=BkUHlMZ0b)| They propose a robustness metric called CLEVER (Cross Lipschitz Extreme Value for nEtwork Robustness). |
|2018| Huan Zhang et al.| [Efficient Neural Network Robustness Certification with General Activation Functions](https://proceedings.neurips.cc/paper/2018/file/d04863f100d59b3eb688a11f95b0ae60-Paper.pdf)|CROWN, a framework to certify robustness of neural networks for given input data points. <br/> Decision of linear and quadratic functions are made according to the activation function. They provide calculation details of tanh, sigmoid, arctan and RELU and claim that the calculations can be generalized. |
| 2019| Hongge Chen et al.| [Robustness Verification of Tree-based Models](https://arxiv.org/abs/1906.03849)| Robustness verification of decision tree ensembles involves finding the exact minimal adversarial perturbation. It can be done  with maximum clique searching algorithms.|
|2019 | Elham Tabassi et al| [A Taxonomy and Terminology of Adversarial Machine Learning](https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8269-draft.pdf)| They present a taxonomy of concepts and define terminology in the field of Adversarial ML. It is arranged in a conceptual hierarchy that includes key types of attacks, defenses, and consequences. |
|2014|Ian Goodfellow et al.|[Explaining and Harnessing Adversarial Examples](https://www.researchgate.net/publication/269935591_Explaining_and_Harnessing_Adversarial_Examples)|They come up with a fast method to generate adversarial examples. They show that "a simple linear model can have adversarial examples if its input has sufficient dimensionality." Their attack works for logistic regression, SVM's and neural networks. |
|2017| Nicholas Carlini et al| [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644)| They introduce a new white-box targeted attack algorithm tailored to three distance metrics.|
| 2017| Alexey Kurakin et al| [Adversarial Examples In The Physical World](https://arxiv.org/abs/1607.02533)| They prove that systems operating in the physical world, using signals from cameras and other sensors as input can be a target for adversarial attacks. While doing so they introduce a new attack called “Basic Iterative Method”.|
| 2016| Seyed-Mohsen Moosavi-Dezfooli et al.| [Deepfool: A Simple and Accurate Method to Fool Deep neural networks](https://arxiv.org/abs/1511.04599)| They claim that they present an accurate method for computing and comparing the robustness of different classifiers. They do it using the DeepFool algorithm with creates adversarial samples.|
|2019|Arjun Nitin Bhagoji et al| [Analyzing Federated Learning through an Adversarial Lens](https://www.research.ibm.com/artificial-intelligence/publications/paper/?id=Analyzing-Federated-Learning-through-an-Adversarial-Lens)| They explore the threat of poisoning attacks on federated learning. They explore several strategies to carry out this attack and try to increase attack stealth.|
|2017| Matthias Hein et al|[Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation](https://www.ml.uni-saarland.de/Publications/HeiAnd-FormGuarAdvManip.pdf)|They propose the Cross-Lipschitz regularization functional. This form of regularization in neural networks improves the robustness of the classifier with no loss in performance.|
|2020|Chuanbiao Song et al| [Robust Local Features For Improving The Generalization of Adversarial Training](https://arxiv.org/pdf/1909.10147.pdf)|They investigate the relationship between the generalization of adversarial training and the robust local features, try to make models learn robust local features from adversarial training.|

<br>

##  Metrics

The main idea of the robustness metrics is to calculate the minimal perturbation that is required to get an input misclassified. In other words to calculate the average sensitivity of the model’s loss function with respect to changes in the inputs.


### Empirical Robustness
Assesses the robustness of a given classifier with respect to a specific attack and test data set. It is equivalent to the average minimal perturbation that the attacker needs to introduce for a successful attack.

To be able to calculate empirical robustness we need to know the type and the parameters of the attack.

### Loss Sensitivity
Local loss sensitivity quantify the smoothness of a model by estimating its Lipschitz continuity constant. Lipschitz constant measures the largest variation of the output of the model under a small change in its input. The smaller the value, the smoother the function.

### CLEVER Score

For a given input CLEVER estimates the minimal perturbation that is required to change the classification. The derivation is based on a Lipschitz constant. The CLEVER algorithm uses an estimate based on extreme value theory.

The CLEVER score can be calculated for both untargeted and targeted attacks. 

A higher CLEVER score indicates better robustness.

#### Loss Sensitivity VS CLEVER Score

CLEVER Score and Loss Sensitivity both uses the same idea of estimating the smoothness by Lipschitz constant. They both are attack-independent. But they have a couple of differences.

* Loss sensitivity uses Local Lipschitz constant estimation meanwhile CLEVER uses Cross Lipschitz constant estimation.

* Even though they try to estimate similar constants, Cross Lipschitz constant calculation includes an additional difference operation. Hence a model with a high CLEVER score would have a low loss sensitivity. For Loss sensitivity; the smaller the value, the smoother the function. For CLEVER score; the greater the value, the more robust the model.

* As CLEVER uses a sampling based method, the scores may vary slightly for different runs while Loss Sensitivity calculation is determined.

### Clique Method Robustness Verification for Decision Tree Ensembles

For decision-tree ensemble models like Gradient Boosted Decision Trees, Random Forest, or Extra Trees robustness can be verified based on clique method.
For our purposes this measure might be too specific.

### CROWN framework for certifying neural networks

CROWN is a framework for efficiently computing a certified lower bound of minimum adversarial distortion given any input data point for neural networks.

To calculate the lower bound one must know about the activation function used in the Neural Network. Then check where the activation function is concave or convex and choose lower/upper bound functions accordingly.

Not a fully designed metric, just gives an upper bound with respect to the given data point.

##  Attacks

As most of the papers use success rate of attacks as a robustness metric we can use state-of-the-art attacks in our algorithm.

### Carlini Wagner attack
It is a white-box targeted attack algorithm tailored to three distance metrics (l<sub>2</sub>, l<sub>0</sub>, l<sub>infinity</sub>). It is referred as a state-of-the-art attack. It tries to optimize a minimization problem using gradient descent.

Quite powerful, however it is often much slower than others.

Even though it is developed for Neural Networks, it can be modified to be effective on Logistic Regression models and Support Vector Classifiers.

### Fast Gradient Attack

It is a white-box attack and has targeted/untargeted versions. Straightforward and practical.
It is effective on Logistic Regression models, Support Vector Classifiers and Neural Networks.

### DeepFool

An efficient attack for deep neural networks. It is white-box and untargeted. For a given input, it finds the nearest decision boundary in l<sub>2</sub> norm. It can be modified to be effective on Logistic Regression models and Support Vector Classifiers.


## Taxonomy

|model type | metric     	| description 	| unit             	|
|-----------|---------------|:-----------:	|------------------	|
|Decision Tree|	Clique Method Robustness Verification|	Gives a lower bound on robustness for decision tree ensembles. Larger value better robustness.| [0 inf]		|
|Neural Network	|Loss Sensitivity|	Quantify the smoothness of a model. Smaller value better robustness.|	[0 inf]	| 
|Neural Network	| CLEVER Score| Estimates the minimal perturbation that is required to change the classification. Higher value better robustness.|	[0 inf]	|
|Neural Network, Logistic Regression, SVM	|Empirical robustness (CW attack)|	Success rate of the CW attack. Smaller value better robustness.	|[0 1]|
|Neural Network, Logistic Regression, SVM	|Empirical robustness (Fast Gradient Method)|	Success rate of the Fast Gradient Attack. Smaller value better robustness.|	[0 1]|
|Neural Network, Logistic Regression, SVM	|Empirical robustness (DeepFool)|	Success rate of the DeepFool attack. Smaller value better robustness.	|[0 1]|


