from modelgym.client import wonderland_pb2
from modelgym.client import wonderland_pb2_grpc
from hashlib import md5
from pathlib import Path
from time import sleep
from shutil import rmtree
from shutil import copyfile
import datetime
import random
import string
import json
import grpc
from os.path import dirname
import os

# Server's ip
ip = 'localhost:50051'
# Folder in Minikube
repoFolder = "/home/docker/repo-storage/test"
homeFolder = os.getenv("HOME") + "/repo-storage/test"


class Client:
    ip = 'localhost:50051'
    crt = ""
    projectFolder = ""
    experimentFolder = ""

    def __init__(self, ip, projectFolder):
        self.ip = ip
        self.projectFolder = projectFolder
        self.channel = grpc.insecure_channel(ip)
        self.stub = wonderland_pb2_grpc.PipelineServiceStub(self.channel)


    def newExperiment(self):
        """
            Creates new folder for an experiment in the project folder.
             Return PATH on the local machine to this folder
        """
        time = datetime.datetime.today()
        experimentFolder = time.strftime("%Y-%m-%d-%H.%M.%S")
        for _ in range(3):
            if not os.path.isdir(self.projectFolder+'/EXPERIMENTS/'+experimentFolder):
                experimentFolder = time.strftime("%Y-%m-%d-%H.%M.%S")
                break
        os.makedirs(self.projectFolder+'/EXPERIMENTS/'+experimentFolder)
        self.experimentFolder = experimentFolder
        return experimentFolder


    def clearExperiment(self):
        """
            Remove current experiment folder
        """
        rmtree(self.projectFolder+'/EXPERIMENTS/'+self.experimentFolder)
        return 0

    def eval(self, model_type, params, datasetPath, metrics, verbose):
        modelpath = self.sendModel(model_type, params, metrics)
        pipe = self.makePipe(modelpath, datasetPath)
        self.stub.LaunchPipeline(pipe)
        output = self.getData(self.projectFolder + '/' + modelpath.split("/")[-2] + "/output.json")
        # results = self.stub.RetrieveResults(models_path)
        opt_metr = output["scores"][0][0]
        return opt_metr

    def eval_by_paths(self, modelPath, dataPath):
        """
            Launch single Pipeline
            :argument
            modelPath: <string>, absolute path to the model on a local machine
            dataPath: <string>, absolute path to the data on a local machine
        """
        pipeline = self.makePipe(modelPath, dataPath)
        self.stub.LaunchPipeline(pipeline)
        return 0


    def makePipe(self, model, data):
        pipe = wonderland_pb2.Pipeline(git_info=wonderland_pb2.GitUrl())

        dataset_data = wonderland_pb2.Dataset(type="path", data=[dirname(data)], container_mount_endpoint="/home/data")
        dataset_model = wonderland_pb2.Dataset(type="path", data=[dirname(model)], container_mount_endpoint="/home/model")
        func = wonderland_pb2.Function(docker_image="training-image",
                                       command_to_execute="trainer",
                                       execution_parameters=["/home/data/" + Path(data).name,
                                                             "/home/model/" + Path(model).name, "/home/model/output.json"],
                                       inputs={"data": dataset_data, "model": dataset_model}
                                       )

        pipe.nodes["1"].func.CopyFrom(func)
        return pipe


    def sendModel(self, model_type, params, metrics):
        model = {"models": [{"type": model_type.__name__,
                             "params": params}],
                 "metrics": [m.name for m in metrics[1:]],
                 "return_models": False
                 }
        folder = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])
        dirpath = self.projectFolder + "/" + folder
        while os.path.isdir(dirpath):
            folder = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])
            dirpath = self.projectFolder + "/" + folder
        os.makedirs(dirpath)
        with open(dirpath + "/model.json", "w") as file:
            json.dump(model, file)

        return repoFolder + "/" + folder + "/model.json"

    def sendData(self, data, tag):
        if type(data) is bytes:
            dataPath = "/tmp/_temp_data"
            with open(dataPath, 'wb') as file:
                file.write(data)
        else:
            dataPath = data
        hash = GetDataHash(dataPath)[:10]
        if hash in [folderName[-10:] for folderName in os.listdir(self.projectFolder+'/DATA')]:
            print("Folder for data already exist!")
            return 0
        time = datetime.datetime.today()
        dataFolder = self.projectFolder+'/DATA/'+time.strftime("%Y-%m-%d-%H.%M")+'#'+hash
        os.makedirs(dataFolder)
        if tag is None:
            filename = Path(dataPath).name
        else:
            filename = tag
        copyfile(dataPath, dataFolder+'/'+filename)

    def getData(self, path):
        while not os.path.exists(path):
            sleep(5)
        data = json.load(open(path))
        return data

def GetDataHash(dataPath):
    BLOCKSIZE = 65536
    hasher = md5()
    with open(dataPath, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

if __name__ == "__main__":
    client = Client('localhost:50051', os.getenv("HOME") + "/repo-storage/test")
    client.sendData(b"/home/igor/dfDownwloads/oleg.rar","data.csv")
    ##experimentFolder = client.newExperiment()

