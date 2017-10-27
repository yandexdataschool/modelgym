IMAGE:=modelgym/dockers:latest
#DOCKERFILE:=environment/Dockerfile.binder
DOCKERFILE:=environment/Dockerfile.ubuntu
CONFIG:=test
NOTEBOOK:=model_search.ipynb
JUPYTER_PORT:=8888
SHELL=/bin/bash
NB_RUNNER=./scripts/run_nb.sh
VERSION=$(shell grep __version__ modelgym/__init__.py | cut -d \" -f 2)

# DISTRIBUTED
NUM_WORKERS:=5
SSH_USER:=anaderi
MAX_FAILURES:=1000
RESERVE_TIMEOUT:=300
POLL_INTERVAL=1.0

include mongo.makefile

test-local: build-base-image
	docker run -i --rm \
	  -v ${DOCKER_ROOT}${CURDIR}:/src \
	  ${IMAGE} \
	  bash --login -c " \
	   cd /src; umask 000 ; ${NB_RUNNER} ${NOTEBOOK} test"

run-fmin-local:
	docker run -i --rm \
	  -v ${DOCKER_ROOT}${CURDIR}:/src \
	  ${IMAGE} \
	  bash --login -c " \
	   cd /src; umask 000 ; ${NB_RUNNER} ${NOTEBOOK} ${CONFIG}"

run-fmin-local-bg:
	docker run -d \
	  -v ${DOCKER_ROOT}${CURDIR}:/src \
	  ${IMAGE} \
	  bash --login -c " \
	   cd /src; umask 000 ; ${NB_RUNNER} ${NOTEBOOK} ${CONFIG}"

run-container-base:
	docker run -ti --rm \
	  -v ${CURDIR}:/src \
	  ${IMAGE} \
	  /bin/bash --login

build-base-image:
	docker build -t ${IMAGE} -f ${DOCKERFILE} environment
	docker tag ${IMAGE} ${IMAGE:latest=${VERSION}}

jupyter:
	if docker ps |grep -q mongo$$ ; then LINK_MONGO='--link mongo:mongo' ; else LINK_MONGO='' ; fi ; \
	echo $$LINK_MONGO ; \
	docker run -d --name jupyter \
	  -p ${JUPYTER_PORT}:8888 \
	  $$LINK_MONGO \
	  -v ${CURDIR}:/notebooks \
	  ${IMAGE}
	@echo "http://$(shell hostname -f):${JUPYTER_PORT}"
	sleep 3
	docker logs jupyter | tail -2

jupyter-stop:
	docker stop jupyter ; docker rm jupyter


##
## DISTRIBUTED
##

test-worker: build-worker-image
	export TARGET_CONFIG=test ; docker run -i --rm \
	  -v ${DOCKER_ROOT}${CURDIR}:/src \
	  ${IMAGE_SKYGRID_WORKER} \
	  bash --login -c " \
	   cd /src; umask 000 ; ${NB_RUNNER} ${NOTEBOOK} $$TARGET_CONFIG"

run-fmin-distributed:
	docker run -i --rm \
	  -v ${DOCKER_ROOT}${CURDIR}:/work \
	  -e MONGO_PORT_27017_TCP_ADDR=${MONGO_HOST} \
	  -e MONGO_PORT_27017_TCP_PORT=${MONGO_PORT} \
	  -e MONGO_DB=${MONGO_DB} \
	  ${IMAGE} \
	  bash --login -c " \
	   cd /work; umask 000 ;${NB_RUNNER} ${NOTEBOOK} ${CONFIG}"

run-worker-local:
	docker run -i --rm \
	  -v ${CURDIR}:/work \
	  ${IMAGE} \
	  bash --login -c "export PYTHONPATH='/work'; \
	  pip install dill; \
	  hyperopt-mongo-worker \
	  --mongo=${MONGO_HOST}:${MONGO_PORT}/${MONGO_DB} \
	  --poll-interval=0.5 \
	  --workdir=/work"

run-worker-docker:
	touch workers.txt
	${CURDIR}/check_jobs.sh
	set -e ; for host in `comm -23 <(sort hosts.txt) <(cut -d ' ' -f 3 workers.txt|sort) | sort -R |head -${NUM_WORKERS}` ; do \
	  TSTAMP=$$(date +"%Y-%m-%d-%H-%M-%S") ; \
	  CONTAINER=$$(ssh -t -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -q ${SSH_USER}@$$host \
	  'set -e ; sudo -u robot-cern-mc-gen docker pull ${IMAGE_SKYGRID_WORKER} ; \
	  sudo -u robot-cern-mc-gen docker run -d \
	  ${IMAGE_SKYGRID_WORKER} \
	  bash --login -c "export PYTHONPATH=/src; \
	  /opt/conda/envs/catboost/bin/python /opt/conda/envs/catboost/bin/hyperopt-mongo-worker \
	  --mongo=${MONGO_HOST}:${MONGO_PORT}/${MONGO_DB} \
	  --poll-interval=${POLL_INTERVAL} \
	  --reserve-timeout=${RESERVE_TIMEOUT} \
	  --max-consecutive-failures=${MAX_FAILURES}"'| tail -1) && \
          echo $$TSTAMP $$(echo $$CONTAINER| cut -c 1-12) $$host | tee -a workers.txt ; \
	done
	
run-worker-skygrid:
	docker run -i --rm \
	  -v ${DOCKER_ROOT}${CURDIR}:/work \
	  ${IMAGE_SKYGRID_CLIENT} \
	  /work/skygrid_worker.py run_worker --num-jobs ${NUM_WORKERS} \
		--mongo-host ${MONGO_HOST} --mongo-port ${MONGO_PORT} --mongo-db ${MONGO_DB} \
		--reserve-timeout ${RESERVE_TIMEOUT} --max-consecutive-failures ${MAX_FAILURES} 

stop-worker-docker:
	${CURDIR}/stop_workers.sh
	${CURDIR}/check_jobs.sh


run-container-skygrid-client:
	docker run -ti --rm \
	  -v ${CURDIR}:/work \
	  ${IMAGE_SKYGRID_CLIENT} \
	  /bin/bash

build-worker-image: # TODO move to environment?
	docker build -f Dockerfile.skygrid-worker -t ${IMAGE_SKYGRID_WORKER} .

build-skygrid-client:
	docker build -t ${IMAGE_SKYGRID_CLIENT} -f Dockerfile.skygrid-client .

push-image:
	docker login -u ${DOCKER_USER} -p ${DOCKER_PASSWORD}
	docker push ${IMAGE_SKYGRID_WORKER}

test:
	pytest tests/ -v -s
