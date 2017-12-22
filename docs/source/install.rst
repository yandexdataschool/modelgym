Installation
====================================

Installation without Docker
------------------------------
**Note:** This installation guide was written for python3

Starting Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create directory where you want to clone this rep and switch to it. Install virtualenv and start it::

    pip3 install virtualenv
    python3 -m venv venv
    source venv/bin/activate

To deactivate simply type ``deactivate``

Installing Dependences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install required python3 packages by running following commands.

1. modelgym::

    pip3 install git+https://github.com/yandexdataschool/modelgym.git

2. jupyter::

    pip3 install jupyter

3. LightGBM. Modelgym works with LightGBM version 2.0.4::

    pip3 intstall lightgbm==2.0.4

4. XGBoost. Modelgym works with XGBoost version 0.6::

    git clone --recursive https://github.com/dmlc/xgboost
    cd xgboost
    git checkout 14fba01b5ac42506741e702d3fde68344a82f9f0
    make -j
    cd python-package; python3 setup.py install
    cd ../../
    rm -rf xgboost

Verification If Model Gym Works Correctly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clone repository::

    git clone https://github.com/yandexdataschool/modelgym.git

Move to example and start jupyter-notebook::

    cd modelgym/example
    jupyter-notebook

Open ``model_search.ipynb`` and run all cells. If there are no errors, everything is allright!

Model Gym With Docker
----------------------

Getting Started
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run model gym inside Docker container you need to have installed
`Docker <https://docs.docker.com/engine/installation/#supported-platforms>`_. Also for Mac or Windows you can install instead `Kitematic <https://kitematic.com>`_.

Download this repo. All of the needed files are in the ``modelgym`` directory::

    $ git clone https://github.com/yandexdataschool/modelgym.git
    $ cd ./modelgym

Running Model Gym In A Container Using DockerHub Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run docker container with official image ``modelgym/jupyter:latest`` from DockerHub repo for using model gym via jupyter you simply run the command::

    $ docker run -ti --rm  -v "$(pwd)":/src  -p 7777:8888 \
    modelgym/jupyter:latest  bash --login -ci 'jupyter notebook'

If you are using Windows you need to run this instead::

    $ docker run -ti --rm  -v %cd%:/src  -p 7777:8888 \
    modelgym/jupyter:latest  bash --login -ci "jupyter notebook"

At first time it downloads container.

Verification If Model Gym Works Correctly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Firstly you should check inside container that ``/src`` is not empty.

To connect to jupyter host in browser check your Docker public ip::

    $ docker-machine ip default

Usually the default ip is ``192.168.99.100``.

When you start a notebook server with token authentication enabled (default), a token is generated to use for authentication. This token is logged to the terminal, so that you can copy it.

Go to ``http://<your published ip>:7777/`` and paste auth token.

Open ``/example/model_search.ipynb`` and try to run all cells. If there are no errors, everything is allright.
