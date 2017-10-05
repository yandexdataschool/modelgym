# model gym
Gym for predictive models

[![run at everware](https://img.shields.io/badge/run%20me-@everware-blue.svg?style=flat)](https://everware.ysda.yandex.net/hub/oauth_login?repourl=https://github.com/yandexdataschool/modelgym)

## model gym with Docker

1. [Getting started](#1-getting-started)  
2. [Running model gym in Docker container](#2-running-model-gym-in-docker-container)

### 1. Getting started
To run model gym inside Docker container you need to have installed
[Docker](https://docs.docker.com/engine/installation/#supported-platforms) (also for Mac and Windows you can install instead [Kitematic](https://kitematic.com)). 

Download this repo. All of the needed files are in the `modelgym` directory.
```sh
$ git clone https://github.com/yandexdataschool/modelgym.git
$ cd ./modelgym
```

### 2. Running model gym in Docker container
In order to run model gym in a single docker containter you can pull model gym Docker image `anaderi/modelgym:0.1.3` from official [repo](https://hub.docker.com/r/anaderi/modelgym/) on DockerHub.

#### Running model gym in a container using DockerHub image
To run docker container with official image `anaderi/modelgym:0.1.3` from DockerHub repo for using model gym you simply run the command:
```sh
$  docker run -ti --rm  -v `pwd`:/src  -p 7777:8888  anaderi/modelgym:0.1.3  bash --login
```
At first time it downloads container.
#### Verification if model gym works correctly

Firstly you should check inside container that `/src` is not empty.

Try to run jupyter notebook and copy auth token:
```sh
$  jupyter notebook
```
To connect to jupyter host in browser go to `http://192.168.99.100:7777/` and paste auth token.

Open `/modelgym/example/model_search.ipynb` and try to run all cells. If there are no errors, everything is allright.
