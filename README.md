# Machine Learning Engineer with Azure
To collect all of project materials from Machine Learning Engineer with Azure from Udacity

## Project 1 - [Optimizing an ML Pipeline in Azure](https://github.com/wasuratme96/MSAzure-ML-Engineer/tree/main/Optimize%20an%20Azure%20ML%20Pipeline)
![png](img/creating-and-optimizing-an-ml-pipeline.png)

In this project, I have created and optimize an ML pipeline by using custom training model script from Logistic Regression and perform hyperparameters tuning by HyperDrive. Comparing with AutoML to build and optimize the model. <br>

Both of pipeline will be compared and tracked on AzureML Studio.
> * **Dataset** <br>
>Bank Marketing from [UCI-ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
> * **Problem Statement** <br>
>To predict will customer subscribe the bank deposit program or not ? (Yes vs. No.) <br>
> * **Model Type** <br>
>Binary Classification


## Project 2 - [Operationalizing ML on azure]()
![png](img/operationalize-architec.png)
This project focus on operationalzation machine learning process on Azure Machine Learning. Starting from model development to create the pipeline of model and monitoring the status of deployed endpoints.

> * **Dataset** <br>
>Bank Marketing from [UCI-ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
> * **Problem Statement** <br>
>To predict will customer subscribe the bank deposit program or not ? (Yes vs. No.) <br>
> * **Tool Inuse** <br>
> Azure Python SDK, Application Insight, Azure container instance, AutoML, Swagger and Apache Benchmark

## Project 3 - [IEEE-CIS Fraud Detection](https://github.com/wasuratme96/MSAzure-ML-Engineer/tree/main/CapStone%20Fraud-Detection)
![ml-ops-diagram](img/pipeline_architec.png)


In this project we will predict the probability of online transation is fradulent.

The data comes from Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features. You also have the opportunity to create new features to improve your results.


*Acknowledgements:*

![vesta](img/vesta-logo.png)

> * **Dataset** <br>
>Vesta Transactional Data ([Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data))
> * **Problem Statement** <br>
>To predict probability of fradulent on each transactional ID.
>*Binary Classification* 
> * **Scoring Metrics** <br>
>Area Under the Curve (AUC)
>*Binary Classification*
> * **Tool Inuse** <br>
> Azure Python SDK, Application Insight, Azure container instance, AutoML, Swagger and Apache Benchmark, HyperDrive for Tuning.

