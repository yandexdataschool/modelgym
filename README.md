# model gym
Gym for predictive models

[![run at everware](https://img.shields.io/badge/run%20me-@everware-blue.svg?style=flat)](https://everware.ysda.yandex.net/hub/oauth_login?repourl=https://github.com/yandexdataschool/modelgym)

## model gym with Docker

1. [Getting started](#1-getting-started)  
2. [Running model gym in a container using DockerHub image](#2-running-model-gym-in-a-container-using-dockerhub-image)
3. [Verification if model gym works correctly](#3-verification-if-model-gym-works-correctly)

### 1. Getting started
To run model gym inside Docker container you need to have installed
[Docker](https://docs.docker.com/engine/installation/#supported-platforms) (also for Mac or Windows you can install instead [Kitematic](https://kitematic.com)). 

Download this repo. All of the needed files are in the `modelgym` directory.
```sh
$ git clone https://github.com/yandexdataschool/modelgym.git
$ cd ./modelgym
```

### 2. Running model gym in a container using DockerHub image
To run docker container with official image `anaderi/modelgym:latest` from DockerHub repo for using model gym via jupyter you simply run the command:
```sh
$  docker run -ti --rm  -v `pwd`:/src  -p 7777:8888 \
   anaderi/modelgym:latest  bash --login -ci 'jupyter notebook'
```
At first time it downloads container.
### 3. Verification if model gym works correctly

Firstly you should check inside container that `/src` is not empty.

To connect to jupyter host in browser check your published ip: `docker-machine ip default` (usually it is 192.168.99.100)

Go to `http://<your published ip>:7777/` and paste auth token.

Open `/example/model_search.ipynb` and try to run all cells. If there are no errors, everything is allright.
