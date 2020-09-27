# This is part of the 'Disaster Response Pipelines' project for the Udacity Data Scientist Specialization.
# The template had been provided as part of the coursework along with some important functionalities in the code.

# General utilities
import sys
import numpy as np
import pickle as pkl
from sqlalchemy import create_engine
import pandas as pd

# Classifier libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# NLTK imports
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('brown')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    category_cols = []
    for col in df.columns:
        if col not in ['id', 'message', 'original', 'genre']:
            category_cols.append(col)

    Y = df[category_cols].values.astype(int)
    
    return X, Y, category_cols


def tokenize(text):
    """
    Function to tokenize input text data. Called in pipeline sentence by sentence.
    Input: text: sentence as string.
    return: clean_tokens: list of clean tokens in sentence.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    Function to build model which involves setting up the pipeline of various steps to train an NLP model.
    Input: None
    Return: model: scikit-learn model: Can be used to train on data and evaluate on test set.
    """
    
    # Steps: tokenize, transform to get Tfidf vectors for data, classifier to train
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Parameter search on sklearn cross validation
    parameters = {
        'clf__estimator__n_estimators': [50,100,200],
        'clf__estimator__learning_rate': [0.1, 0.5, 1.0]
    }

    # grid search on data to obtain best parameters.
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return model


def get_results(category_cols, y_test, preds):
    """
    Function to calculate f1-score, precision and recall of each category.
    Uses classification report function of scikit learn and reports the macro average which is the non-weighted
    average of each category.
    Input: category_cols: category columns in data as list
           y_test: Labels as numpy array
           preds: Output predictions from classifier
    Output: dict: results dictionary from classification report function.
    """
    results = dict()
    avg_calc = {'f1_score': [], 'precision': [], 'recall': []}
    print("Category : f1-score, precision, recall\n")
    for ix, col in enumerate(category_cols):
        results[col] = classification_report(y_test[:,ix], preds[:, ix], output_dict=True)
        f1_score, precision, recall = results[col]['macro avg']['f1-score'], results[col]['macro avg']['precision'], \
                                      results[col]['macro avg']['recall']
        avg_calc['f1_score'].append(f1_score)
        avg_calc['precision'].append(precision)
        avg_calc['recall'].append(recall)
        print("{} : {}, {}, {}".format(col, f1_score, precision, recall))
    
    print("\n\n\n")
    print("Avg. across categories:")
    print("f1-score: {}".format(np.mean(avg_calc['f1_score'])))
    print("precision: {}".format(np.mean(avg_calc['precision'])))
    print("recall: {}".format(np.mean(avg_calc['recall'])))
    return results


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate model on test data.
    Input: model: scikit-learn model: Mode trained on train data. Should have a .predict functionality.
            X_test: array/numpy array: Test set from train_test_split function.
            Y_test: array/numpy array: Labels for test set from train_test_split function.
    Return: None
    """
    preds = model.predict(X_test)
    results = get_results(category_names, Y_test, preds)
    
    return


def save_model(model, model_filepath):
    """
    Function to save model to local filepath.
    Input: model: scikit-learn model: Mode trained on train data. Should have a .predict functionality.
           model_filepath: str: containing the path to the model.
    Return: None
    """
    pkl.dump(model, open(model_filepath, 'wb'))
    
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
