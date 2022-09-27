#import packages
import sys
import re 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """loads data from different files then merges it 
    Args:
    messages_filepath: file path to 'messages.csv'
    categories_filepath: file path to 'categories.csv'

    Returns :
    df : DataFrame contains data from two merged dataframes
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df


def clean_data(df):
    """cleans the dataframe by :
               1- creating 36 different columns to be input for the ML model
               2- cleaning their names
               3- converting those 36 columns into dummy columns (only 0s and 1s)
               4- checking for duplicated rows and dropping them
    Args:
    df : DataFrame contains data from two merged dataframes

    Returns :
    df : cleaned DataFrame 
    """
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [row[i][0].split('-')[0] for i in row]
    categories.columns = category_colnames
    for column in categories:
         # set each value to be the last character of the string
            categories[column] = categories[column].str.split('-').str[1]
    
         # convert column from string to numeric
            categories[column] = categories[column].astype(int)
    
    #dropping the original categories column
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    df.replace({'related':{2: 1}},inplace=True)#making the 'related' column only contain 0s and 1s by removing 2s from it
    return df





def save_data(df, database_filename):#save data to a database
    sql = 'sqlite:///{}'.format(database_filename)
    engine = create_engine(sql)
    df.to_sql('message categories', engine, index=False,if_exists='replace')  




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