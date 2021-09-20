# Optimizing an Azure ML Pipeline

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

## Summary
This data set is Bank Marketing from [UCI-ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). <br>
The data is related with direct marketing campaigns of a Portuguese banking institution. This marketing campaigns is conducted by phone calls. Purpose is to direct contact to customer if the product (bank term deposit) would be `('yes')` or  `('no')` in subscription. <br>
So, we want to use these historical data features to predict whether bank term doposit would be `('yes')` or `('no')` in subscription.

The best performing model is a `VotingEnssemble` from AutoML Pipeline with accuracy = **0.91549** comparing with **0.91002** in accuracy from `Scikit Learn-Logisitc Regression with HyperDrive`

## ML Pipeline
### Scikit-learn Pipeline with HyperDrive
#### Pipeline Architecture
On Scikit-lean pipeline, it will compose with python script (`train.py`) to handle all the data preprocessing and training Logistic Regression Model from Scikit-Learn. Then Jupyter Notebook (`udacity-project.ipynb`) will be used to orchestrate all the process with Azure ML environment via Azure SDK. 

-> Starting from connect to the **Workspace** 

-> create **ComputeTarget** 

-> config **Environment** for python script 

-> establish and config **Hyperparameter Tuning** with **HyperDrive** 

-> submit **Experiment** to Azure ML 

-> register best model.



### AutoML

### Pipeline comparison

## Future Work

## Reference

### Initial code
[Udacity](https://github.com/udacity/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files)

### AzureML SDK
[Tune Hyperparameters](https://github.com/MicrosoftLearning/mslearn-dp100/blob/main/11%20-%20Tune%20Hyperparameters.ipynb) <br>
[Use Automated Machine Learning](https://github.com/MicrosoftLearning/mslearn-dp100/blob/main/12%20-%20Use%20Automated%20Machine%20Learning.ipynb)
