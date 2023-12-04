# DEMO: Interpreting Many-Objective Feature Selection (MOFS) Solutions
In this repository, you will find the codebase for the experiments of our proposed method for the interpretability of MOFS results.
Given a dataset, certain preprocessing is necessary to get it fit for feature selection. This includes dealing with missing and outlier values and normalising the data.
With the preprocessed dataset, we proceed with MOFS and then present the results through a dashboard that facilitates interpretability, as shown in the figure below.

![MOFS system overview](systemoverview.png)

To perform MOFS, we consider two cases in this work: four and six objectives.
## Prerequisites
In other to run the scripts, prerequisite libraries in **requirements.txt** file must be installed. Before doing so, I recommend setting up a [Virtual Environment (VE)](https://docs.python.org/3/library/venv.html) for this experiment, after which you install the required libraries in the VE with the following command:
```
pip install -r requirements.txt
```
## Four objectives
Three arguments are required to execute the ```four_objectives.py``` script:
- file path: this is the path to the file for execution.
- target feature: the name of the target feature.
- classifier: classifier to use for the experiment. This could be a Decision tree (dt), Logistic regression (lr) or, if not specified, Naive Bayes (NB).
To execute the script, use the command below with the arguments specified.
```
python four_objectives.py [file path] [target feature] ["dt"|"lr"]
```
## Six objectives
In addition to the three arguments needed above, the ```six_objectives.py``` script requires a fourth one:
- sensitive feature: this is the sensitive attribute on which fairness should be achieved.
To execute the script, use the command below with the arguments specified.
```
python six_objectives.py [file path] [target feature] ["dt"|"lr"] [sensitive feature]
```
