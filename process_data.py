# This is part of the 'Disaster Response Pipelines' project for the Udacity Data Scientist Specialization.
# The template had been provided as part of the coursework along with some important functionalities in the code.

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to load data given disaster messages and corresponding categories file.
    Input: messages_filepath: str, containing the path to the messages file.
           categories_filepath: str, containing the path to the categories file.
    Return: df: pandas DataFrame: Merged data of both the above.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """
    Function to clean data before training a model and testing for evaluation.
    Input: df: pandas DataFrame: Input data with messages and cateogories.
    return: df: pandas DataFrame: Clean data
    """
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row]
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column].astype(str))

    # drop the original categories column from `df`
    df.drop(labels='categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)
    # drop duplicates
    df.drop_duplicates(subset='id', inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Function to save data to local database file.
    Input: df: pandas DataFrame: Input data with messages and cateogories.
           database_filename: str, paht to save the db file.
    return: None
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)
    
    return 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
