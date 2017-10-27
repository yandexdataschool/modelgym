# Model Gym
Gym for predictive models

[![run at everware](https://img.shields.io/badge/run%20me-@everware-blue.svg?style=flat)](https://everware.ysda.yandex.net/hub/oauth_login?repourl=https://github.com/yandexdataschool/modelgym)

## What is this about?

Modelgym is a place (a library?) to get your predictive models as meaningful in a smooth and effortless manner. Modelgym provides the unified interface for 
- different kind of Models (XGBoost, CatBoost etc) that you can tune with
- different kind of optimization strategies, or Trainers. 
Usually it starts with a data sample that you can be evaluate with our Guru interface and maybe get a few hints on preprocessing it before starting the training. 
In the end of the trials you get a report of the model performance on both cross-validation and hold-out samples. So you can pick the top model along the best hyperparameters for your case. 

**Installation**
1. [Starting Virtual Environment](#1-starting-virtual-environment)
2. [Installing Dependences](#2-installing-dependences)
3. [Verification If Model Gym Works Correctly](#verify-1)

**Installation With Docker**
1. [Getting Started](#1-getting-started)
2. [Running Model Gym In A Container Using DockerHub Image](#2-running-model-gym-in-a-container-using-dockerhub-image)
3. [Verification If Model Gym Works Correctly](#verify-2)


## Installation
**Note:** This installation guide was written for python3
### 1. Starting Virtual Environment
Create directory where you want to clone this rep and switch to it. Install virtualenv and start it.
```sh
pip3 install virtualenv
python3 -m venv venv
source venv/bin/activate
```
To deactivate simply type ```deactivate```
### 2. Installing Dependences
Install required python3 packages by running following commands.
1. modelgym
    ```sh
    pip3 install git+https://github.com/yandexdataschool/modelgym.git
    ```
2. jupyter, yaml, hyperopt, skopt, pandas and networkx
    ```sh
    pip3 install jupyter pyyaml hyperopt scikit-optimize pandas networkx==1.11
    ```
3. LightGBM. **Note:** Modelgym works with LightGBM version 2.0.2
    ```sh
    git clone --recursive https://github.com/Microsoft/LightGBM
    cd LightGBM
    git checkout 80c641cd17727bebea613af3cbfe3b985dbd3313
    mkdir build && cd build && cmake -DUSE_MPI=ON ..
    make -j
    cd ../python-package/ && python3 setup.py install
    cd ../../
    rm -rf LightGBM
    ```
4. XGBoost. **Note:** Modelgym works with XGBoost version 0.6.
    ```sh
    git clone --recursive https://github.com/dmlc/xgboost
    cd xgboost
    git checkout 14fba01b5ac42506741e702d3fde68344a82f9f0
    make -j
    cd python-package; python3 setup.py install
    cd ../../
    rm -rf xgboost
    ```
### <a name="verify-1"></a> 3. Verification If Model Gym Works Correctly
Clone repository.
```sh
git clone https://github.com/yandexdataschool/modelgym.git
```
Move to example and start jupyter-notebook.
```sh
cd modelgym/example
jupyter-notebook
```
Open ```model_search.ipynb``` and run all cells. If there are no errors, everything is allright!

## Model Gym With Docker
### 1. Getting Started
To run model gym inside Docker container you need to have installed
[Docker](https://docs.docker.com/engine/installation/#supported-platforms) (also for Mac or Windows you can install instead [Kitematic](https://kitematic.com)).

Download this repo. All of the needed files are in the `modelgym` directory.
```sh
$ git clone https://github.com/yandexdataschool/modelgym.git
$ cd ./modelgym
```

### 2. Running Model Gym In A Container Using DockerHub Image
To run docker container with official image `modelgym/dockers:latest` from DockerHub repo for using model gym via jupyter you simply run the command:
```sh
$  docker run -ti --rm  -v "$(pwd)":/src  -p 7777:8888 \
   modelgym/dockers:latest  bash --login -ci 'jupyter notebook'
```
At first time it downloads container.
### <a name="verify-2"></a> 3. Verification If Model Gym Works Correctly

Firstly you should check inside container that `/src` is not empty.

To connect to jupyter host in browser check your Docker public ip:
```sh
$ docker-machine ip default
```
(usually it's 192.168.99.100)

When you start a notebook server with token authentication enabled (default), a token is generated to use for authentication. This token is logged to the terminal, so that you can copy it.

Go to `http://<your published ip>:7777/` and paste auth token.

Open `/example/model_search.ipynb` and try to run all cells. If there are no errors, everything is allright.
