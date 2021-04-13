
# Trusted AI

## Project Documentation 

The whole documentation of the project can be found on our [GitHub page](https://joelleupp.github.io/Trusted-AI/)

## Time Schedule

- [ ] State of the art of pillars and metrics (12-Mar)
  - [x] Fairness
  - [ ] Explainability
  - [ ] Robustness
  - [ ] Transparency
- [ ] Identify missing aspects relevant for calculating AI trust score (26-Mar)
- [ ] Create taxonomy (2-Apr)
- [ ] Select an application scenario (9-Apr)
- [ ] Design a trusted AI algorithm (7-Mai)
- [ ] Implementation (18-Jun)
- [ ] Evaluation and Discussion (9-Jul)
- [ ] Documentation and Process (30-Jul)


### mkdocs

The documentation is created with mkdocs, which uses simple markdown file and a .yml config file to build and deploy the documentation to GitHub.

The relevant files to create the documentation are in the [mkdocs folder](https://github.com/JoelLeupp/Trusted-AI/tree/main/mkdocs)

The file mkdocs.yml is the config file and has to be adapted if the theme should change or the index changes or new markdowns added. 
The markdown files are in the folder docs and include the whole documentation.

installation

```sh
pip install --upgrade pip
pip install mkdocs
pip install pymdown-extensions
pip install python-markdown-math
```

workflow:

1.  At the location of the mkdocs folder open the terminal and enter:

        
        mkdocs serve
        

    This will build the documentation and host it locally, where the full documentation can be seen.

2.  change the config or markdown files. Sinc you are serving the documentation locally you can see the changes in the config or markdown files in real time.
You will always see the output and how the documentation will look in the end and therefore minimalize syntax errors in markdown for example. 

3.  deploy the build on GitHub to the branche gh-pages. For this open another termianl at the localtion of the mkdocs folder and enter:
        
        mkdocs gh-deploy
        
4. commit and push the changed config or markdown files (keep the source of the build up to date)
