#!/bin/bash
set -e

apt-get install -y vim netcat iputils-ping

conda install -y jupyter matplotlib seaborn runipy
conda clean -tipsy

# install jupyterhub
ENV_NAME=$CONDA_DEFAULT_ENV
source deactivate
pip install jupyterhub==0.7.* notebook==5.0.*
source activate $ENV_NAME

echo "Generating jupyter config"
jupyter notebook -y --generate-config --allow-root
cat << EOL_CONFIG >> $HOME/.jupyter/jupyter_notebook_config.py
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True
EOL_CONFIG


echo "Registering environment $ENV_NAME as kernel for jupyterhub and jupyter"
ipython kernel install --name $ENV_NAME

# changing matplotlib configuration
mkdir -p $HOME/.config/matplotlib && echo 'backend: agg' > $HOME/.config/matplotlib/matplotlibrc
