# model gym
Gym for predictive models

[![run at everware](https://img.shields.io/badge/run%20me-@everware-blue.svg?style=flat)](https://everware.ysda.yandex.net/hub/oauth_login?repourl=https://github.com/yandexdataschool/modelgym)


## Installation
**Note:** This gym works with python3
1. Create directory where you want to clone this rep and switch to it.
2. Install virtualenv and start it.
    ```
    pip3 install virtualenv
    python3 -m venv venv
    source venv/bin/activate
    ```
    To deactivate simply type ```deactivate```
3. Clone repository:
    ```
    git clone https://github.com/yandexdataschool/modelgym.git
    ```
4. Install required python3 packages:
    1. modelgym:
	```
	pip3 install git+https://github.com/yandexdataschool/modelgym.git
	```
    2. jupyter, yaml, hyperopt, skopt, pandas and networkx:

	```
	pip3 install jupyter pyyaml hyperopt scikit-optimize pandas networkx==1.11
	```
    3. lightGBM:
	```
	git clone --recursive https://github.com/Microsoft/LightGBM
	cd LightGBM
	git checkout 80c641cd17727bebea613af3cbfe3b985dbd3313
	mkdir build && cd build && cmake -DUSE_MPI=ON ..
	make -j
	cd ../python-package/ && python3 setup.py install
	cd ../../
	rm -rf LightGBM
	```
    4. XGBoost:
	```
	git clone --recursive https://github.com/dmlc/xgboost
	cd xgboost
	git checkout 14fba01b5ac42506741e702d3fde68344a82f9f0
	make -j
	cd python-package; python3 setup.py install
	cd ../../
	rm -rf xgboost
	```
5. Verify that the installation was successful.
    1. Move to example and start jupyter-notebook:
	```
	cd modelgym/example
	jupyter-notebook
	```
    2. Open ```model_search.ipynb```.
    3. Run all cells. If there are no errors, everything is allright!
