# Code for Pys-TaLiRo


# Installation

We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

After you've installed poetry, you can install partx by running the following command in the root of the project: 

```
poetry install
```

After installing, you will probbaly face a matlab error. 
base don your matlab version, choose a correct matlabengine version from [https://pypi.org/project/matlabengine/#history](https://pypi.org/project/matlabengine/#history). 
Run:


# What to look at?

1) To run the demo: 
    Check if the system works and produces correct robustness values based on your box, requirements. 

    ```
        cd demos
        poetry run python demo1.py
    ```

2) Run demo2 to run the falsifier. Note that the boxes are for not phi.

    ```
        cd demos
        poetry run python demo2.py
    ```