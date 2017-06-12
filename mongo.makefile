
#MONGO
MONGO_DB=trials_ecal_${CONFIG}
MONGO_PORT:=8270
MONGO_HOST:=cern-mc01h.ydf.yandex.net


mongo:
	mkdir -p ${CURDIR}/db
	docker run -d --name mongo \
	  -v ${CURDIR}/db:/data/db \
	  -p ${MONGO_PORT}:27017 \
	  mongo 

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

