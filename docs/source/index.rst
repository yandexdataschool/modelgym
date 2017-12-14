.. modelgym documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. we can avoid copypasting this and installation guide from README
   when m2r add end-line and start-line arguments to their mdinclude

Model Gym
====================================

What is this about?
====================================

Modelgym is a place (a library?) to get your predictive models as meaningful in a smooth and effortless manner. Modelgym provides the unified interface for

* different kind of Models (XGBoost, CatBoost etc) that you can tune with
* different kind of optimization strategies, or Trainers.

Usually it starts with a data sample that you can be evaluate with our Guru interface and maybe get a few hints on preprocessing it before starting the training.
In the end of the trials you get a report of the model performance on both cross-validation and hold-out samples. So you can pick the top model along the best hyperparameters for your case.

Installation
==================
.. toctree::
    :maxdepth: 3

    install

Examples
==================
.. toctree::
   :maxdepth: 3

   train_example
   guru_example

Documentaion
==================
.. toctree::
   :maxdepth: 2

   guru
   models
   trainer
   tracker
