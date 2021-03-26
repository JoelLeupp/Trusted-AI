# Trusted AI Master Project 

This site serves as documentation and Overview of the Master Project.

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



