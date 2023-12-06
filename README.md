# DEMO: Interpreting Many-Objective Feature Selection (MOFS) Solutions
In this repository, you will find the codebase for the experiments of our proposed method for the interpretability of MOFS results.
Given a dataset, certain preprocessing is necessary to get it fit for feature selection. This includes dealing with missing and outlier values and normalising the data.
With the preprocessed dataset, we proceed with MOFS and then present the results through a dashboard that facilitates interpretability, as shown in the figure below.

![MOFS system overview](systemoverview.png)

To perform MOFS, we consider two cases in this work: four and six objectives.
## Prerequisites
To run the scripts, prerequisite libraries in the ```requirements.txt``` file must be installed. Before doing so, I recommend setting up a [Virtual Environment (VE)](https://docs.python.org/3/library/venv.html) for this experiment, after which you install the required libraries in the VE with the following command:
```
pip install -r requirements.txt
```
## Execution
The python file ```methodology.py``` takes in a preprocessed dataset, executes MOFS, and then returns three files:
* The file with the set of solutions and their corresponding objective values.
* An image with the individual feature contrition to the classification problem using the SHAP measure.
* A spreadsheet with three sheets: one for the objective ranks, another for the rank of the solutions using TOPSIS, and the third with the frequency of the individual features.
These can then be used to create a dashboard like the one in this demonstration to facilitate choosing a final solution from the set of solutions.
Below, we present the commands for executing the **methodology.py** file for four and six objective problems.
* ### Four objectives
Three arguments are required to execute the ```methodology.py``` script for four objectives:
- file path: this is the path to the file for execution.
- target feature: the name of the target feature.
- classifier: classifier to use for the experiment. This could be a Decision tree (dt), Logistic regression (lr) or, if not specified, Naive Bayes (NB).
- weight method: there are three options for this argument: entropy, equal, or rs (which is the range/STD for each objective)
To execute the script, you can just use the command below with the arguments specified.
```
python methodology.py [file path] [target feature] ["dt"|"lr"] ["equal"|"entropy"|"rs"]
```
* ### Six objectives
For six objectives, the required arguments to execute ```six_objectives.py``` are:
- file path: this is the path to the file for execution.
- target feature: the name of the target feature.
- classifier: classifier to use for the experiment. This could be a Decision tree (dt), Logistic regression (lr) or, if not specified, Naive Bayes (NB).
-  sensitive feature: this is the sensitive attribute on which fairness should be achieved.
- weight method: there are three options for this argument: entropy, equal, or rs (which is the range/STD for each objective)
To execute the script, you can just use the command below with the arguments specified.
```
python methodology.py [file path] [target feature] ["dt"|"lr"] [sensitive feature] ["equal"|"entropy"|"rs"]
```

