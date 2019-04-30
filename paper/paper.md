---
title: 'Modelgym – an optimizer for decision tree models!'
tags:
  - machine learning
  - decision-tree based models
authors:
 - name: Andrey Ustyuzhanin
   orcid: 0000-0001-7865-2357
   affiliation: "1, 2"

 - name: Sergey Miller
   orcid: 0000-0003-4613-1199
   affiliation: "3"

 - name: Fedor Minkin
   orcid: 0000-0002-3353-6560
   affiliation: "1, 3"

 - name: Alexander Boymel
   orcid: 0000-0003-0759-6831
   affiliation: "1, 3"
 
 - name: Alexey Zabelin
   orcid: 0000-0003-4654-4495
   affiliation: "3"
   
affiliations:
 - name: Yandex
   index: 1
 - name: Yandex Data Factory
   index: 2
 - name: Moscow Institute of Physics and Technology
   index: 3
date: 17 December 2017
---

# Summary

The decision-tree based models in machine learning have gained prominence in the last decades since they are widely applicable, relatively fast and particularly useful for heterogeneous (i.e. real-world) input data. However, at the current point in the development of these computational tools, one must be intimately familiar with specific frameworks, their tunable parameters and features to get the most out of the available data. Modelgym lowers the entry barrier to solving complex <classification and> regression problems by introducing a concise, easy-to use interface for training, tuning and evaluation of decision tree models across different frameworks.
Modelgym provides Guru, Model, Trainer and Tracker classes which respectively cover data preparation, model building, model selection and optimization tracking tasks. Guru allows detection of correlated, categorical and sparse features in the dataset; Model provides a unified interface for model building, supporting gradient boosting frameworks LightGBM, XGBoost and <Catboost> and Random Forest implementation from scikit-learn; Trainer tunes hyperparameters of different Model objects using a selection of optimizers, currently Tree of Parzen Estimators from hyperopt and Bayesian optimization using Gaussian Processes from skopt. The intermediate results of model optimization are saved by Tracker either to a file or to a MongoDB.
Since installation and compilation of abovementioned frameworks can itself be a challenge, Modelgym is distributed both as source code on Github and as a pre-built Docker image on Docker Hub, making its features accessible on Linux, Windows and MacOS even to an inexperienced data scientist.

# References

[C.Spearman. 1904. The Proof and Measurement of Association between Two Things.](http://www.jstor.org/stable/1412159?origin=JSTOR-pdf&seq=1#page_scan_tab_contents)

[Robert E. Banfield, Kevin W. Bowyer, Lawrence O. Hall, W.P. Kegelmeyer. 2007. A Comparison of Decision Tree Ensemble
Creation Techniques.](http://www.sandia.gov/~wpk/pubs/publications/pami06.pdf)

[Hyperopt documentation.](http://hyperopt.github.io/hyperopt/)

[Scikit-optimize documentation.](https://scikit-optimize.github.io/#skopt.forest_minimize)

[XGBoost documentation.](http://xgboost.readthedocs.io/en/latest/)

[Lightgbm documentation.](http://lightgbm.readthedocs.io/en/latest/)

[Sklearn documentation.](http://scikit-learn.org/stable/index.html)

