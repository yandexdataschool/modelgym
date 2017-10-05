# model gym
Gym for predictive models

[![run at everware](https://img.shields.io/badge/run%20me-@everware-blue.svg?style=flat)](https://everware.ysda.yandex.net/hub/oauth_login?repourl=https://github.com/yandexdataschool/modelgym)

1. [Getting started](#1-getting-started)  
2. [Running model gym in Docker container](#2-running-model-gym-in-docker-container)

### 1. Getting started

???

### 2. Running model gym in Docker container
Firstly install [docker](https://docs.docker.com/engine/installation/#supported-platforms) (also for Mac and Windows you can install instead [Kitematic](https://kitematic.com)).
In order to run model gym in a single docker containter you can pull model gym Docker image `anaderi/modelgym:0.1.3` from official [repo](https://hub.docker.com/r/anaderi/modelgym/) on DockerHub.

#### Running model gym in a container using DockerHub image
To run model gym with official image `anaderi/modelgym:0.1.3` from DockerHub repo you simply run the command:
```sh
$  docker run -ti --rm  -v `pwd`:/src  -p 7777:8888  anaderi/modelgym:0.1.3  bash --login
```



