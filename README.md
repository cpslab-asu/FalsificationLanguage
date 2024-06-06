# Code for Pys-TaLiRo


# Installation

We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

After you've installed poetry, you can install partx by running the following command in the root of the project: 

```
poetry install
```

# What to look at?

1) `falsi_lang.new_monitor.Requirement` : 
    
    - Implemented the abstract class `staliro.core.Specification` from `psy-taliro` package
    - Refer to [https://sbtg.gitlab.io/psy-taliro/specifications.html#specification](https://sbtg.gitlab.io/psy-taliro/specifications.html#specification)
        - I have followed the same structure - `falsi_lang.new_monitor.Requirement.evalute` takes in two inputs: states and times and return the robustness values.

2) To run the demo: 

    ```
        cd demos
        poetry run python test_monitor.py
    ```
