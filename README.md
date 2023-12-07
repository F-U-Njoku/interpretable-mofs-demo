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
1. The file with the set of solutions and their corresponding objective values.
2. An image with the individual feature contrition to the classification problem using the SHAP measure.
3. A spreadsheet with three sheets:
    1. one for the objective ranks,
    2. another for the rank of the solutions using TOPSIS,
    3. and the third with the frequency of the individual features.
  
These can then be used to create a dashboard like the one in this demonstration to facilitate choosing a final solution from the set of solutions.
Below, we present the commands for executing the ```methodology.py``` file for four and six objective problems.
* ### Four objectives
The four objectives considered are _subset size, accuracy, F1 score, and Variance Inflation Factor (VIF)_. The required arguments to execute the ```methodology.py``` script for four objectives are:
1. file path: this is the path to the file for execution (acceptable file format is CSV).
2. target feature: the name of the target feature.
3. classifier: classifier to use for the experiment. This could be a Decision tree (dt), Logistic regression (lr) or, if not specified, Naive Bayes (NB).
4. weight method: there are three options for this argument: entropy, equal, or rs (which is the range/STD for each objective)
To execute the script, you can just use the command below with the arguments specified.
```
python methodology.py [file path] [target feature] ["dt"|"lr"] ["equal"|"entropy"|"rs"]
```
* ### Six objectives
The six objectives considered are _subset size, accuracy, F1 score, VIF, statistical parity, and equalised odds_. The required arguments to execute the ```methodology.py``` script for six objectives are:
1. file path: this is the path to the file for execution (acceptable file format is CSV).
2. target feature: the name of the target feature.
3. classifier: classifier to use for the experiment. This could be a Decision tree (dt), Logistic regression (lr) or, if not specified, Naive Bayes (NB).
4.  sensitive feature: this is the sensitive attribute on which fairness should be achieved.
5. weight method: there are three options for this argument: entropy, equal, or rs (which is the range/STD for each objective)
To execute the script, you can just use the command below with the arguments specified.
```
python methodology.py [file path] [target feature] ["dt"|"lr"] [sensitive feature] ["equal"|"entropy"|"rs"]
```

