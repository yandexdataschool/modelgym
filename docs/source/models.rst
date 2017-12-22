Models
===========

In order to use our Trainer you need the wrapper on your model.
You can find the required `Model interface`_ below.

We implement wrappers for several models:

* `XGBoost`_
* `LightGBM`_
* `RandomForestClassifier`_
* `Catboost`_

Also, we implement an `Ensemble Model`_.

Model interface
---------------
.. automodule:: modelgym.models.model
    :members:
    :undoc-members:

XGBoost
---------------
.. automodule:: modelgym.models.xgboost_model
    :members:
    :undoc-members:
    :show-inheritance:


LightGBM
---------------
.. automodule:: modelgym.models.lightgbm_model
    :members:
    :undoc-members:
    :show-inheritance:

RandomForestClassifier
------------------------------
.. automodule:: modelgym.models.rf_model
    :members:
    :undoc-members:
    :show-inheritance:

Catboost
--------
.. automodule:: modelgym.models.catboost_model
    :members:
    :undoc-members:
    :show-inheritance:

Ensemble Model
--------------
.. automodule:: modelgym.models.ensemble_model
    :members:
    :undoc-members:
    :show-inheritance:
