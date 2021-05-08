ml_project
==============================

First homework on course "ML in production". As an example task was taken "SMS spam detection"

Link: https://www.kaggle.com/uciml/sms-spam-collection-dataset


What it is?
--------
ALl project consist of 4 stages:
1) Preparation Data
   
   Texts and label columns are selected from datasets. Tokenizing and creating vocabulary. 
   It is possible to use pretrained vectors in vocabulary (like GloVe)
   
2) Train model
   
   It is main stage. Here train process is performed. A lot of parameters are configurable. 
   Like model (only rnn is implemented by now, layers, input/hidden size etc), epochs, 
   learning rate, step size (if step scheduler is used) and so on. Look at config.py detaily
   
3) Prediction
   
   Use previously pretrained model to make prediction.

4) Visualization
   
   Finally, a short report is created via streamlit. A page is created where you can see a short 
   insight on dataset and train history (loss, score functions by epochs). Example of such report 
   is saved in pdf file (reports/visualize Streamlit.pdf)
   
Configuration is done by yaml files (folder configs) and hydra framework.

Data 
----
Dataset was taken from kaggle: https://www.kaggle.com/uciml/sms-spam-collection-dataset/data

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── tests              <- Tests for projects
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Tests for processing from raw data to processed
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
