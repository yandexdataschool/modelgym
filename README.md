# Model Gym
Gym for predictive models

[![run at everware](https://img.shields.io/badge/run%20me-@everware-blue.svg?style=flat)](https://everware.ysda.yandex.net/hub/oauth_login?repourl=https://github.com/yandexdataschool/modelgym)

## Model Gym With Docker

1. [Getting Started](#1-getting-started)  
2. [Running Model Gym In A Container Using DockerHub Image](#2-running-model-gym-in-a-container-using-dockerhub-image)
3. [Verification If Model Gym Works Correctly](#3-verification-if-model-gym-works-correctly)

### 1. Getting Started
To run model gym inside Docker container you need to have installed
[Docker](https://docs.docker.com/engine/installation/#supported-platforms) (also for Mac or Windows you can install instead [Kitematic](https://kitematic.com)). 

Download this repo. All of the needed files are in the `modelgym` directory.
```sh
$ git clone https://github.com/yandexdataschool/modelgym.git
$ cd ./modelgym
```

### 2. Running Model Gym In A Container Using DockerHub Image
To run docker container with official image `anaderi/modelgym:latest` from DockerHub repo for using model gym via jupyter you simply run the command:
```sh
$  docker run -ti --rm  -v "$(pwd)":/src  -p 7777:8888 \
   modelgym/dockers:latest  bash --login -ci 'jupyter notebook'
```
At first time it downloads container.
### 3. Verification If Model Gym Works Correctly

Firstly you should check inside container that `/src` is not empty.

To connect to jupyter host in browser check your Docker public ip: 
```sh 
$ docker-machine ip default
``` 
(usually it's 192.168.99.100)

When you start a notebook server with token authentication enabled (default), a token is generated to use for authentication. This token is logged to the terminal, so that you can copy it.

Go to `http://<your published ip>:7777/` and paste auth token.

Open `/example/model_search.ipynb` and try to run all cells. If there are no errors, everything is allright.
