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

```
docker build -t bankchurn .

docker run -it -p 8000:8000 bankchurn
```

## 2. Create IAM user

## 3. Create ECR repository

## 4. Creaet EC2 instance

Can be ubuntu

## 5. Setup EC2 instance

```
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

#In case of error
sudo chmod 666 /var/run/docker.sock

```

## 6. Create a runner in github

### Download

#### Create a folder

```
mkdir actions-runner && cd actions-runner# Download the latest runner package
curl -o actions-runner-linux-x64-2.313.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.313.0/actions-runner-linux-x64-2.313.0.tar.gz
echo "56910d6628b41f99d9a1c5fe9df54981ad5d8c9e42fc14899dcc177e222e71c4 actions-runner-linux-x64-2.313.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.313.0.tar.gz
```

### Configure

#### Create the runner and start the configuration experience

```
./config.sh --url https://github.com/luis95garay/data_science_bank_churn --token A2HZ3RAIEHR54EP7JI4PWY3F2JIFM
./run.sh
```

### Using your self-hosted runner

The name must be self-hosted

## 7. Create github credentials

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>> id.dkr.ecr.us-east-1.amazonaws.com

ECR_REPOSITORY_NAME = bankchurn

## 7. Add and push the github workflow

## 8. Update security group

Add custom TCP in port 8000

## 9. Display in EC2

Remember to replace https to http

## 10. Clean resources

- Delete app runner
- Terminate EC2
- Delete ECR repository
