# Bank Churn

This is a project of bank churn prediction developed in kedro framework. The project is devided in two pipelines

1. data_processing: It performes steps like: remove unnecesary columns, handle outliers, treat skewed columns, one hot encoding and feature selection
2. data_science: It performes steps like: split data, train and evaluate model

## Overview

This is a Kedro project, which was generated using `Kedro 0.17.7`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## How to install dependencies

Create an environment and make sure that it is with python 3.6

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run Kedro project

If we want to run all the processing and training steps, run:

```
kedro run
```

If we only want to run a specific pipeline run:

```
kedro run --pipeline=data_science
```

If we want to visualize the final results run:

```
kedro viz --autoreload
```

# Model Deployment

## 1. Validate docker images

## 2. Create IAM user

## 3. Create ECR repository

## 4. Creaet EC2 instance ubuntu

## 5. Setup EC2 instance

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

## 6. Create a runner in github

## 7. Create github credentials

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>> id.dkr.ecr.us-east-1.amazonaws.com

ECR_REPOSITORY_NAME = bankchurn

## 7. Push the githubworkflow
