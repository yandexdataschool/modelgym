IMAGE:=anaderi/modelgym:latest
DOCKERFILE:=environment/Dockerfile.ubuntu
CONFIG:=test
NOTEBOOK:=model_search.ipynb
JUPYTER_PORT:=8888
SHELL=/bin/bash
NB_RUNNER=./scripts/run_nb.sh

#MONGO
MONGO_DB=trials_ecal_${CONFIG}
MONGO_PORT:=8270
MONGO_HOST:=cern-mc01h.ydf.yandex.net

# DISTRIBUTED
NUM_WORKERS:=5
SSH_USER:=anaderi
MAX_FAILURES:=1000
RESERVE_TIMEOUT:=300
POLL_INTERVAL=1.0

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

jupyter:
	docker run -d --name jupyter -p ${JUPYTER_PORT}:8888 -v ${CURDIR}:/notebooks ${IMAGE}
	@echo "http://$(shell hostname -f):${JUPYTER_PORT}"
	sleep 3
	docker logs jupyter | tail -2

jupyter-stop:
	docker stop jupyter ; docker rm jupyter

##
## MONGO
##

mongo:
	docker run -d --name mongo -p ${MONGO_PORT}:27017 mongo 

mongo-stop:
	docker stop mongo
	docker rm mongo

mongo-drop-db:
	docker exec -i mongo mongo ${MONGO_DB} --eval "db.dropDatabase()"

mongo-reset-jobs:
	docker exec -i mongo mongo ${MONGO_DB} --eval 'db.jobs.updateMany({$$and: [ {state: {$$eq: 1}}, {tid: {$$gt: 102}}]}, {$$set: {owner: null, state: 0} });'

mongo-express:
	docker run -d -p 8081:8081 --name mongo-express \
	  --link mongo:mongo \
	  -e ME_CONFIG_MONGODB_ENABLE_ADMIN=true \
	  -e ME_CONFIG_OPTIONS_EDITORTHEME="ambiance" \
	  mongo-express
	echo "http://${MONGO_HOST}:8081"

mongo-express-stop:
	docker stop mongo-express
	docker rm mongo-express

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

