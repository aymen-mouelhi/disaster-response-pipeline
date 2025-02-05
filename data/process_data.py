'''
ETL Pipeline that consists of three major parts:
* Loading 2 datasets and merging them based on id
* Cleaning the categories and removing duplicates
* Storing the cleaned DataFrame into an sqlite database
'''


import sys
import pandas as pd
import numpy as np
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    """
    Load data from the csv files and merges them in a DataFrame
    Args:
        messages_filepath: path of the messages csv file
        categories_filepath: path of the categories csv file
    Returns:
        (DataFrame) df: Merged DataFrame of messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='inner', on=['id'])

    return df


def _get_processed_categories(df):
    """
    Return the dataframe categories
    Args:
        df: Merged  DataFrame
    Returns:
        categories: list of categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int).apply(lambda x: 1 if x>=2 else x)

    return categories


def clean_data(df):
    # process categories
    categories = _get_processed_categories(df)
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    duplicated = df.duplicated()
    duplicated_count = len(duplicated[duplicated == True])

    # drop duplicates
    df.drop_duplicates(inplace=True, keep='first')

    return df

def save_data(df, database_filename):
    """
    Save clean dataset into an sqlite database
    Args:
        df:  Cleaned dataframe
        database_filename: Name of the database file
    """
    engine = sqlalchemy.create_engine('sqlite:///'+database_filename)
    df.to_sql('categorized_messages', engine, index=False)


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
