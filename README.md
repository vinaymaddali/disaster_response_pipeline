# disaster_response_pipeline

# Installation
This code runs python 3.
Jupyter notebook was installed using Anaconda: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/install.html <br />
XGBoost package: https://xgboost.readthedocs.io/en/latest/get_started.html
Pandarallel: https://pypi.org/project/pandarallel/

# Data
The data files were provided in the project by Udacity. It consists of messages sent during various disaster events
along with a labeled category for each message. The following are included in the repository:
1. messages.csv
2. categories.csv

# Project motivation
This article is an assignment for the Udacity data scientist specialization. The task is to use data engineering and NLP
techniques to read, clean and categorize data into the appropriate classes in order to send the message to the
right disaster relief agency.

# Files
1. process_data.py: Loads data (messages and categories), cleans, merges and saves to a local db.
2. train_classifier.py: Tokenizes and extracts features on data, performs a parameter search using cross validation on
                        train data, builds and saves model that classifies categories of messages.
3. run.py: Brings up a web application that visualizes data and uses saved model from train_classifier.py to classify
           input messages.
4. ModelComparison.ipynb: Similar to train_classifier.py but provides an extended analysis with performances of other
                          classifiers and features.

# How to bring up application
python process_data.py <messages.csv/(messages file)> <categories.csv/(categories file)> DisasterResponse.db
python train_classifier.py DisasterResponse.db classifier.pkl
python run.py
To bring up web app, access: http://0.0.0.0:3001/

# Results
Refer to ModelComparison.ipynb for comparison of different classifiers and a word2vec feature extractor.

# References:
https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html

# Acknowledgements
Thanks to Udacity for this project and pointing me to this dataset. <br />
Thanks to FigureEight and the contributors for the data.