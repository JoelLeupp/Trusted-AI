# Trusted AI Master Project 

This site serves as documentation and Overview of the Master Project.

## Introduction

Artificial intelligence (AI) systems are getting more and more relevance as support to
human decision-making processes. While AI holds the promise of delivering valuable insights
into many application scenarios, the broad adoption of AI systems will rely on the ability to
trust their decisions. According to IBM, some key aspects provide information to trust (or not)
a decision made by an algorithm. In particular, a decision must: be reliable and fair, be
accounted for, not cause harm, not be tampered with, be understandable, and be
secure. In this context, the previous aspects have been grouped into the following pillars.

• Fairness. One of the essential aspects to enable trusted AI is to avoid bias across
the entire lifecycle of AI applications, as well as across different bias and data types.

• Robustness. AI-based systems may be vulnerable to adversarial attacks. Attackers
may poison training data by injecting samples to compromise system decisions eventually.

• Explainability. In many application scenarios, decisions made by algorithms must be
explainable and understandable by humans. Various stakeholders require explanations for
different purposes, and explanations must be tailored to their needs.

• Accountability. The quality of decisions should be evaluated in terms of accuracy,
ability to satisfy users’ preferences, as well as other properties related to the impact of the
decision.The previous characteristics are demanded from AI systems. Yet, to achieve trust in AI, more
efforts and automatic mechanisms able to calculate and communicate trust levels are required.

## Project Documentation 

The whole documentation of the project can be found on our [GitHub page](https://joelleupp.github.io/Trusted-AI/)


### mkdocs

The documentation is created with mkdocs, which uses simple markdown file and a .yml config file to build and deploy the documentation to GitHub.

The relevant files to create the documentation are in the [mkdocs folder](https://github.com/JoelLeupp/Trusted-AI/tree/main/mkdocs)

The file mkdocs.yml is the config file and has to be adapted if the theme should change or the index changes or new markdowns added. 
The markdown files are in the folder docs and include the whole documentation.

installation

```sh
pip install --upgrade pip
pip install mkdocs
```

workflow:

1.  At the location of the mkdocs folder open the terminal and enter:

        
        mkdocs serve
        

    This will build the documentation and host it locally, where the full documentation can be seen.

2.  change the config or markdown files. Sinc you are serving the documentation locally you can see the changes in the config or markdown files in real time.
You will always see the output and how the documentation will look in the end and therefore minimalize syntax errors in markdown for example. 

3.  deploy the build on GitHub to the branche gh-pages. For this open another termianl at the localtion of the mkdocs folder and enter:
        
        mkdocs gh-deploy



