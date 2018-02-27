from modelgym.client import wonderland_pb2
from modelgym.client import wonderland_pb2_grpc
from hashlib import md5
from pathlib import Path
from time import sleep
from shutil import rmtree
from shutil import copyfile
from os.path import dirname
import datetime
import random
import string
import json
import grpc
import os

from modelgym.models import CtBClassifier
from modelgym.trainers import TpeTrainer
from modelgym.utils import ModelSpace
from hyperopt import hp
from modelgym.models import LGBMClassifier
from modelgym.metrics import Accuracy

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
        self.newExperiment()


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
        os.makedirs(dirname(self.projectFolder)+'/EXPERIMENTS/'+experimentFolder)
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
        output = self.getData(dirname(self.projectFolder) + '/' + modelpath + "/output.json")
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


    def makePipe(self, modelPath, dataPath):
        """
        Create message for Wonderland grpc-server.
          Parameter `data` of wonderland_pb2.Dataset contains part of path on a remote server
        :param model: <string>. Model's folder name.
        :param data:  <string>. Data's folder name.
        :return: wonderland_pb2.Pipeline
        """
        pipe = wonderland_pb2.Pipeline(git_info=wonderland_pb2.GitUrl())

        dataset_data = wonderland_pb2.Dataset(type="path",
                                              data=[dataPath],
                                              container_mount_endpoint="/home/data")
        dataset_model = wonderland_pb2.Dataset(type="path",
                                               data=[modelPath],
                                               container_mount_endpoint="/home/model")
        func = wonderland_pb2.Function(docker_image="training-image",
                                       command_to_execute="trainer",
                                       execution_parameters=["/home/data/data.csv",
                                                             "/home/model/model.json", "/home/model/output.json"],
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
        folder = 'model-' + ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
        dirpath = self.projectFolder + "/EXPERIMENTS/" + self.experimentFolder + '/' + folder
        for _ in range(3):
            if os.path.isdir(dirpath):
                folder = 'model-' + ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
                dirpath = self.projectFolder + "/EXPERIMENTS/" + self.experimentFolder + '/' + folder
        os.makedirs(dirpath)
        with open(dirpath + "/model.json", "w") as file:
            json.dump(model, file)

        return Path(self.projectFolder).name+"/EXPERIMENTS/"+self.experimentFolder +'/'+folder

    def sendData(self, data):
        """
        Copy data to the local DATA directory, that must be mounted with Azure FS.
        (can be replaced by rpc method)

        :param data: <bytes> or <string>. Specify you data path by string
        or send binary data directly.
        :return: Folder's name.
        """

        if type(data) is bytes:
            dataPath = "/tmp/data.csv"
            with open(dataPath, 'wb') as file:
                file.write(data)
        else:
            dataPath = data
        hash = GetDataHash(dataPath)[:10]
        for folderName in os.listdir(self.projectFolder+'/DATA'):
            if hash == folderName[-10:]:
                print("Folder for data already exist!")
                return Path(self.projectFolder).name+'/DATA/'+folderName
        time = datetime.datetime.today()
        dataFolder = Path(self.projectFolder).name+'/DATA/'+time.strftime("%Y-%m-%d-%H.%M")+'#'+hash
        os.makedirs(dirname(self.projectFolder) + dataFolder)
        copyfile(dataPath, dirname(self.projectFolder) + dataFolder+"/data.csv")
        return dataFolder

    def getData(self, path):
        while not os.path.exists(path):
            sleep(5)
        data = json.load(open(path))
        return data

def GetDataHash(dataPath):
    """
    Calculate md5 hash of data file

    :param dataPath: <string>, data's path
    :return: <string>
    """
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
    ##experimentFolder = client.newExperiment()
    dataPath = client.sendData("/home/igor/cobrain/src/gitlab.com/cobrain-core/training-image/trainer/sample_data.csv")
    print(dataPath)
    ##Select models and spaces

    models = ModelSpace(CtBClassifier,
                        space={'learning_rate': hp.loguniform('learning_rate', -5, -1)},
                        space_update=True)

    trainer = TpeTrainer(models)

    ##Non-cluster optimization
    trainer.crossval_optimize_params(Accuracy(), dataPath, metrics=[Accuracy()], client=client)






