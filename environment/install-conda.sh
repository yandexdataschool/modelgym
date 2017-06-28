#!/bin/bash
set -e

function throw_error {
  echo -e $*
  exit 1
}

if ! which conda ; then
    export CONDA_PATH=/opt/conda
    echo "installing miniconda root environment with jupyterhub"
    MINICONDA_FILE="Miniconda3-4.3.14-Linux-x86_64.sh"
    wget http://repo.continuum.io/miniconda/$MINICONDA_FILE -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $CONDA_PATH || throw_error "Error installing miniconda"
    rm ./miniconda.sh
    export PATH=$CONDA_PATH/bin:$PATH
    hash -r
    conda update --yes conda
else
    export CONDA_PATH=$(dirname $(dirname $(which conda)))
fi

ENV_FILE="environment.yaml"
ENV_NAME="python3"

if ! conda env list|grep  $ENV_NAME ; then
    echo "Creating conda venv $ENV_NAME"
    conda env create -q --file $ENV_FILE -n $ENV_NAME
    conda clean -tipsy
else
    conda env update -f $ENV_FILE -n $ENV_NAME
fi
conda env list
cat >> ~/.profile << EOT  
export PATH=$CONDA_PATH/bin:\$PATH
source activate $ENV_NAME
EOT

source activate $ENV_NAME

# XGBoost
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
git checkout 14fba01b5ac42506741e702d3fde68344a82f9f0
make -j
cd python-package; python setup.py install
cd ../../
rm -rf xgboost

# LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM
git checkout d12e5e4c74b2a0b23ddc49df41ce4deaf02612d2
mkdir build && cd build && cmake -DUSE_MPI=ON ..
make -j
cd ../python-package/ && python setup.py install
cd ../../
rm -rf LightGBM

echo "Python version:"
which python
python --version

