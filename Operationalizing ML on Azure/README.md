# Operationalizing Machine Learning on Azure

## Overview
This is 2nd project from Udacity - Machine Learning Engineering with Azure.<br/>
In this project, ```AutoML``` from Azure have been used to train and find the model then deploy it to the endpoints for consume.<br/>
Endpoints status have been tracked by ```Applications Insight service``` from Azure and ```Swagger``` is used for API documentations. After that all of these process have been created as a publish pipeline on Azure ML Studio

Below is simple workflow for this project.
![workflow](img/operationalize-step-of-work.png)
**Credit : Udacity - Machine Learning Engineer with Azure**

## Summary of Working Process
>### Step 1 : Authentication
>`az cli` is used to log in to the `Azure ML Studio` and >Service Pricipal have been created to access the project >workspace.

>### Step 2 : AutoML Model Training
>This step AutoML from Azure use to train classification models >on [Bank Marketing UCI Data](https://automlsamplenotebookdata.>blob.core.windows.net/automl-sample-notebook-data/>bankmarketing_train.csv) and set the target performance >metrics to be **AUC weighted accuracy**. 

>### Step 3 : Deploy best model performance
>After complete AutoML process, best performer will be deployed >via Azure ML Studio with Azure Container Instance compute >type. Model endpoints will be generated.

>### Step 4 : Enable application insight and maek API >documentations
>Use 'az cli' to enable application insight (tracking reposnse >time, number of request etc.) loging and status of endpoints >can be monitored from Azure Application Insight Service.

>### Step 5 : Consume Model Endpoints (Testing)
>Use 'az cli' to make the request to endpoint via HTTP post >request and recieve the reponse from the endpoint. Data for >testing on consume endpoints also saved at ```data.json```

>### Step 6 : Create and Publish Pipeline
>To make the process of training AutoML can be re-usable, >training AutoML pipeline will be created and published on >Azure Machine Learning Studio.

>### Step 7 : Documentation and Screencast
>Make the summary and documentation (screen capture, screen >cast) create README.md

## Architecture Diagram
![architecture](img/operationalize-architec.png)
Their are 3 mains sub-process in this project
- **AutoML Development**
- **AzureML Pipeline Automation**
- **Model Deployment**

## 1. AutoML Development
### 1.1) Dataset
In this project we use the same dataset as previouse one which is [Bank Marketing UCI Data](https://automlsamplenotebookdata.>blob.core.windows.net/automl-sample-notebook-data/>bankmarketing_train.csv). 

Target from this dataset is to predict whether customer will except the subsciption offer or not (yes. vs no.) base on 4 groups of features 

- *Client features*
- *Campaign Features - Current* 
- *Campaign Features - Previous*
- *Social & Economics Features* 

 Below is python script to register dataset into Azure Machine Learning workspace 
 ```python
found = False
key = "BankMarketing Dataset"
description_text = "Bank Marketing DataSet for Udacity Course 2"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)


df = dataset.to_pandas_dataframe()
```
After run the script, dataset will be registerd on datasets menu as below pictures.
![dataset](img/AutoMLExperiment/registered-dataset.png)

### 1.2) AutoML Setting
AutoML is set as below python script. Task is ```classification```, primary performance metrics is ```AUC_weighted```. Early stopping also be used here. 

```python
automl_settings = {
    "experiment_timeout_hours": 1,
    "max_concurrent_iterations": 3,
    "primary_metric" : 'AUC_weighted'
}

automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="y",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```
Then output and input path have been established at workspace default datastore.
```python
from azureml.pipeline.core import PipelineData, TrainingOutput

ds = ws.get_default_datastore()
metrics_output_name = 'metrics_output'
best_model_output_name = 'best_model_output'

metrics_data = PipelineData(name='metrics_data',
                           datastore=ds,
                           pipeline_output_name=metrics_output_name,
                           training_output=TrainingOutput(type='Metrics'))
model_data = PipelineData(name='model_data',
                           datastore=ds,
                           pipeline_output_name=best_model_output_name,
                           training_output=TrainingOutput(type='Model'))
```
And create AutoMLStep as a one step of pipeline.
```python
automl_step = AutoMLStep(
    name='automl_module',
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True)
```
### 1.3) Create pipeline and submit experiment
Create pipeline from AutoML step object and sugmit the experiment to run.
```python
from azureml.pipeline.core import Pipeline

pipeline = Pipeline(
    description="pipeline_with_automlstep",
    workspace=ws,    
    steps=[automl_step])

pipeline_run = experiment.submit(pipeline)
```

### 1.4) Pipeline Running and Status Monitoring

![[pipelinerun]](img/AutoMLExperiment/automl-pipeline-run.png)