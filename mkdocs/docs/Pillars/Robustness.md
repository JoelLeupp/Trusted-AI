# Robustness

## Introduction
Machine Learning models might not be robust against adversarial perturbations. With visually imperceptible adversarial perturbations, attackers may cause an input to be misclassified. These attacks can be generated in both the black-box setting, where the parameters of the model are unknown to the attacker, and the white-box settings, where attacker have the all necessary information about the model. To trust AI we need to be sure that it is robust against adversarial attacks.

## Key Takeaways, Questions and Limitations
* *Key Takeaways*
    * Robustness is a mathematical term and context independent.
* *Limitations*
    *  By definition we can only talk about robustness of classifiers.
    *  Robustness only makes sense if the input to the model is sufficiently large. When the input is adversarially perturbed the difference should be unnoticiable to the human eye but still be large enough to make the input misclassified. This is only possible when the input is high dimensional so that the differences can be hidden. Thats why they usually use Convolutional Neural Networks as an example.
* *Open Questions*
    * How should our algorithm act on regression models?
    *   The topic is well explored on neural network classifiers but there are almost zero sources on different models. How should our algorithm evaluate different classifier models?
## Sources
### Websites 

* IBM Robustness: [Robustness main page](https://www.research.ibm.com/artificial-intelligence/publications/?researcharea=robustness)

* Adversarial Robustness 360: [Toolbox](http://art360.mybluemix.net/)

* Adversarial Robustness Toolbox Github repo: [ART v1.5](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

* CLEVER Github repo: [Robustness Score](https://github.com/IBM/CLEVER-Robustness-Score)

### Papers

* [Adversarial Robustness Toolbox v1.0.0](https://arxiv.org/pdf/1807.01069.pdf)

* [Evaluating The Robustness of Neural Networks: An Extreme Value Theory Approach](https://openreview.net/pdf?id=BkUHlMZ0b)

* [Efficient Neural Network Robustness Certification with General Activation Functions](https://proceedings.neurips.cc/paper/2018/file/d04863f100d59b3eb688a11f95b0ae60-Paper.pdf)

* [Robust Local Features For Improving The Generalization of Adversarial Training](https://arxiv.org/pdf/1909.10147.pdf)

* [Analyzing Federated Learning through an Adversarial Lens](https://www.research.ibm.com/artificial-intelligence/publications/paper/?id=Analyzing-Federated-Learning-through-an-Adversarial-Lens)

* [Formal Guarantees on the Robustness of a Classifier against Adversarial Manipulation](https://www.ml.uni-saarland.de/Publications/HeiAnd-FormGuarAdvManip.pdf)


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

### Basic Iterative Method

It is an extension of the fast gradient attack algorithm. It is a black-box attack and has targeted/untargeted versions. Straightforward and practical.

### DeepFool

An efficient attack for deep neural networks. It is black-box and untargeted. For a given input, it finds the nearest decision boundary in l<sub>2</sub> norm.

## Taxonomy

|model type | metric     	| description 	| unit             	| weight 	|
|-----------|---------------|:-----------:	|------------------	|--------	|
|Decision Tree|	Clique Method Robustness Verification|	Gives a lower bound on robustness for decision tree ensembles. Larger value better robustness.| [0 inf]		|100%|
|Neural Network	|Loss Sensitivity|	Quantify the smoothness of a model. Smaller value better robustness.|	[0 inf]	| 20% |
|Neural Network	| CLEVER Score| Estimates the minimal perturbation that is required to change the classification. Higher value better robustness.|	[0 inf]	|20%|
|Neural Network	|Empirical robustness (CW attack)|	Success rate of the CW attack. Smaller value better robustness.	|[0 1]|	20%|
|Neural Network	|Empirical robustness (Basic Iterative Method)|	Success rate of the Basic Iterative Method. Smaller value better robustness.|	[0 1]|	20%|
|Neural Network	|Empirical robustness (DeepFool)|	Success rate of the DeepFool attack. Smaller value better robustness.	|[0 1]|	20%|


