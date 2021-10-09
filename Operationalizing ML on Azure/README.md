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
>This step AutoML from Azure use to train classification models on [Bank Marketing UCI Data](https://automlsamplenotebookdata.>blob.core.windows.net/automl-sample-notebook-data/>bankmarketing_train.csv) and set the target performance metrics to be **AUC weighted accuracy**. 

>### Step 3 : Deploy best model performance
>After complete AutoML process, best performer will be deployed >via Azure ML Studio with Azure Container Instance compute >type. Model endpoints will be generated.

>### Step 4 : Enable application insight and make API documentation.
>Use 'az cli' to enable application insight (tracking reposnse >time, number of request etc.) loging and status of endpoints >can be monitored from Azure Application Insight Service.

>### Step 5 : Consume Model Endpoint (Testing)
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

## 1. Authentication
To interact with Azure Machine Learnin Studio from Python SDK in local machine, we need to login to Azure Account via Azure CLI. It will pop-up the login page and request to input usename and password of your Azure account.
```python
az login 
```
After that create the Service Principal(SP) and allow SP to access my workspace
```python
az ad sp create-for-rbac --sdk-auth --name  ml-auth
```
Then we can check the available SP in resource group by using below script.
In the --display-name, it require to input your resource group name.
```python
az ad sp list --display-name azure-ml-eng-udacity
```
![authen](img/AutoMLExperiment/service-pricipal-creation.png)
## 2.) AutoML Model Training
### 2.1) Dataset
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

### 2.2) AutoML Setting
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
### 2.3) Create pipeline and submit experiment
Create pipeline from AutoML step object and sugmit the experiment to run.
```python
from azureml.pipeline.core import Pipeline

pipeline = Pipeline(
    description="pipeline_with_automlstep",
    workspace=ws,    
    steps=[automl_step])

pipeline_run = experiment.submit(pipeline)
```

### 2.4) Pipeline Running and Status Monitoring
Python Azure SDK also have the widget to monitor running status by usinng `RunDetails` on object `Piepline`.
```python
from azureml.widgets import RunDetails
RunDetails(pipeline_run).show()
```
After experiment is completed the widget will shown `complete` status as below picture.
![[pipelinecompleted]](img/AutoMLExperiment/pipeline-experiment-completed.png)
Pipeline status also can be checked from Azure Machine Learning Studio.
![[pipelinecompleted]](img/AutoMLExperiment/automl-experiment-completed.png)

## 3.) Deploy best model performance
### 3.1) Trained model list
 After AutoML Experiment completed, we can check the trained model from below menu in Azure Machine Learning Studio.
 ![[best-model-section]](img/AutoMLExperiment/best-model.png)
The list of trained model ranking by performance is shown as below.
![[model-list]](img/AutoMLExperiment/trained-model-list.png)
In this case, I get `VotingEnsemble` as the best model at 0.94728 'AUC weighted` score.

### 3.2) Deploy best model
This process can be done by menu on Azure Machine Learning Studio. Click on `Deploy` in experiment with best model .The model was deployed by Azure Container Instance (**ACI**) with **Enable Authetication** 
![[model-deploy]](img/ModelDeployment/azi-model-deployment.png)

 Wait until the deployment process is completed. Deployed model can be founded on `Endpoints` menu (left side tabs)
 ![[model-endpoint]](img/ModelDeployment/model-endpoint.png)

## 4.) Enable application insight and make API documentation.
### 4.1) Application Insight Service

After model was deployed, next step is to enable the application insight service from Azure Machine Learning by using python script in `log.py`. In the script we need to indicate the correct model deployed name and use `config.json` file from our Azure Machine Learning Workspace.

`config.json`
```python
{
    "subscription_id": "697bfbda-adf3-4ec7-8411-3d929a4099d8",
    "resource_group": "ml-eng-generic-dev",
    "workspace_name": "azure-ml-eng-udacity"
}
```

`log.py`
```python
from azureml.core import Workspace
from azureml.core.webservice import Webservice

# Requires the config to be downloaded first to the current working directory
ws = Workspace.from_config()

# Set with the deployment name
name = "best-model-deploy"

# load existing web service
service = Webservice(name=name, workspace=ws)

service.update(enable_app_insights=True)

logs = service.get_logs()

for line in logs.split('\n'):
    print(line)
```
Then the status of Application Insight will be changed as below picture.
![[app-insight]](img/ModelDeployment/application-insight-url.png)

### 4.2) API Endpoints Documentation (Swagger)
Swagger is built-in tool that helps to document and consume RESTful web service. <br/>
Azure provides a swagger.json that is used to create a web site that documents the HTTP endpoint for a deployed model.

To make the local host for swagger stagging. `swagger.sh` and `serve.py` is need to start by putting `swagger.json` from Azure Machine Learning Studio in the same directory. User interface after host in the local would be as below.
![swagger-url-ui](img/Swagger/swagger-url-ui.png)

The HTTP API methods and responses for deployed model are as in the picture.
![swagger-api](img/Swagger/swagger-api-example.png)

## 5.) Consume Model Endpoint (Testing)
### 5.2) Consume endpoint
To consumne the model we can use HTTP post request to the endpoint by specify the model uri (`scorint_url`) and autheticate key(`key`) from Azure Machine Learning Studio. 
![endpoint](img/ConsumeEndpoint/model-endpoint-azureml.png)

In this project `endpoint.py` will be use to test the endpoint.

2 Examples input data for prediction have been created according to the format from Swagger HTTP post example. 
```python
import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://a283c576-f785-4d84-8148-5e035a1d4fe9.southeastasia.azurecontainer.io/score'

# If the service is authenticated, set the key or token
key = '5JxhysUDljNl2HyXNTH4eOFfVhMKx40O'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 17,
            "campaign": 1,
            "cons.conf.idx": -46.2,
            "cons.price.idx": 92.893,
            "contact": "cellular",
            "day_of_week": "mon",
            "default": "no",
            "duration": 971,
            "education": "university.degree",
            "emp.var.rate": -1.8,
            "euribor3m": 1.299,
            "housing": "yes",
            "job": "blue-collar",
            "loan": "yes",
            "marital": "married",
            "month": "may",
            "nr.employed": 5099.1,
            "pdays": 999,
            "poutcome": "failure",
            "previous": 1
          },
          {
            "age": 87,
            "campaign": 1,
            "cons.conf.idx": -46.2,
            "cons.price.idx": 92.893,
            "contact": "cellular",
            "day_of_week": "mon",
            "default": "no",
            "duration": 471,
            "education": "university.degree",
            "emp.var.rate": -1.8,
            "euribor3m": 1.299,
            "housing": "yes",
            "job": "blue-collar",
            "loan": "yes",
            "marital": "married",
            "month": "may",
            "nr.employed": 5099.1,
            "pdays": 999,
            "poutcome": "failure",
            "previous": 1
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
```

After execute the `endpoint.py` prediction results of 2 example data are shown as below
![predicted_result](img/ConsumeEndpoint/endpoints-example.png)

### 5.2) Endpoint Benchmarking by Apache Benchmark
After we ensure that model endpoint can be requested, benchmarking on the endpoint will be performed by using [Apache Benchmarking](https://httpd.apache.org/docs/2.4/programs/ab.html)

`benchmark.sh` will be use to run for benchmarking which need to speficify model uri (`scorint_url`) and autheticate key(`key`)

```python
ab -n 10 -v 4 -p data.json -T 'application/json' -H 'Authorization: Bearer 5JxhysUDljNl2HyXNTH4eOFfVhMKx40O' http://a283c576-f785-4d84-8148-5e035a1d4fe9.southeastasia.azurecontainer.io/score
```

Afte executed, result woul be shown as below picture.
![benchmark](img/ConsumeEndpoint/benchmarking.png)
You can see the this script perform 10 request to endpoints and zero fail requests is achieved and average of response time per request is 193.613 ms. By the ways, this test have been conducted in closed environment. When make use the endpoint in real-application environment. Response time per request can be expected to slower than this. 

## 6.) Create and Publish Pipeline
In final part, to make the whole process be able to rerun base on condition or scheduled we will create the pipeline object in Azure Machine Learning Studio and publish as `RESTEndpoint` to make it be able to interact (HTTP Trigger) from other serivces.

### 6.1) Pipeline Creation
To create pipeline, we use `Python Azure SDK` as below.
Where `pipeline_run` is completed pipeline running from AutoML  experiment section (We can seperate the AutoML experiment into specific python script but here we use the best model that got from the 1st run of completed pipeline)
```python
published_pipeline = pipeline_run.publish_pipeline(
    name="Bankmarketing Train", description="Training bankmarketing pipeline", version="1.0")
```
After published, `Pipeline endpoints` will shown up in Azure Machine Learning Studio on `Pipeline` menu.
![pipeline-deployment](img/PipelineDeployment/pipeline-initial-deployment.png)

We can get REST endpoints when click into the deployed pipeline.
![pipeline-endpoint](img/PipelineDeployment/pipeline-endpoint.png)

### 6.2) Make POST request to pipeline endpoint
Same as best model endpoint, we can send the HTTP POST request to trigger the pipeline to run by using python script below.

This time, due to I use the Jupyter-notebook in Azure Machine Learning Studio. To make the Authenitcationm we can function from `azureml.core.authentication`

```python
import requests
from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "pipeline-rest-endpoint"}
                        )
```
When make the request, experiment of pipeline will be create and running status will be shown as belows from Jupter-Notebook in Azure Machine Learning Studio.

![pipeline-status-sdk](img/PipelineDeployment/pipeline-run-notbooksdk.png)
And we also can check the running status from the Azure Machine Learning Studio.
![pipeline-status-amlstudio](img/PipelineDeployment/pipeline-endpoints-completed.png)

(Ps. Due to their are nothings change in the pipeline and I have set AutoMLStep to allow reuse, so running status after make the request will change to completed immediately)
>```python
>automl_step = AutoMLStep(
>    name='automl_module',
>    automl_config=automl_config,
>    outputs=[metrics_data, model_data],
>    allow_reuse=True)
>```

# Future Improvement
- Add cleaning data step in pipeline to make data cleaner.
- Fine tune AutoML by using explanabilit function to explore the feature importance.
- Add model validation step in pipeline to prevent model over -fitting and under-ftting
- Seperate the process/ task in pipeline that can be parallel.
- 
# ScreenCast
Below is screencst the entire process of working with Machine Learning Application that completed within this project.
<a href="https://youtu.be/7G9dQ-RzDvY" target="_blank"><img src="img/screencast-cover.png"
alt="Operationalizing ML on Azure Project" width="240" height="180" border="10" /></a>

# Citations
[Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-macos
)

[Create Sevice Principal](https://docs.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli)

[Machine Learning Pipeline](https://github.com/Azure/MachineLearningNotebooks/blob/master/tutorials/machine-learning-pipelines-advanced/tutorial-pipeline-batch-scoring-classification.ipynb)

[Publish ML Pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-pipelines#publish-a-pipeline)