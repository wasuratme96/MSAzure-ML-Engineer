# Operationalizing Machine Learning on Azure

## Overview
This is 2nd project from Udacity - Machine Learning Engineering with Azure.<br/>
In this project, ```AutoML``` from Azure have been used to train and find the model then deploy it to the endpoints for consume.<br/>
Endpoints status have been tracked by ```Applications Insight service``` from Azure and ```Swagger``` is used for API documentations. After that all of these process have been created as a publish pipeline on Azure ML Studio

Below is simple workflow for this project.
![workflow](img/operationalize-step-of-work.png)
**Credit :Udacity**

## Step 1 : Authentication
`az cli` is used to log in to the `Azure ML Studio` and Service Pricipal have been created to access the project workspace. Below is shown the Service Pricipal list in my Azure ML workspace.

![png](img/AutoMLExperiment/service-pricipal-creation.png)


## Step 2 : AutoML Model Training
This step AutoML from Azure use to train classification models on this [Bank Marketing UCI Data](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) and present the trained models in descending order of **AUC weighted accuracy**.

## Key Steps
*TODO*: Write a short discription of the key steps. Remeber to include all the screenshots required to demonstrate key steps. 

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
