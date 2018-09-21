# Disaster Response Pipeline Project

### Introduction:

This project aims at analyzing and categorizing messages sent during disasters to better organize help for victims. It is composed of an ETL Pipeline, an ML pipeline and a Flask web application.

#### ETL Pipeline
This pipelins loads the different csv files and then cleans the data and store it in an SQLite Database. The steps of this pipeline are:
* Loading 2 datasets and merging them into a global Pandas DataFrame.
* Cleaning the categories and removing duplicates.
* Storing the cleaned DataFrame into an sqlite database.

#### ML Pipeline
This pipeline handles the training of a Multi-class machine learning model and stores the final model in a pickle file.
The steps of this pipeline are:
* Loading the data stored in the sqlite database
* Building a machine learning pipeline based on CountVectorizer, tfidf and AdaBoostClassifier
* Tuning the pipeline using GridSearch
* Evaluating the pipeline and printing a classification report
* Sqving the final model into a pickle file

#### Flask Web Application
Basically, the web application uses the trained model (stored in the pickle file) to classify the entered messages.
Application Graphs:
- Distribution of Message Genres
- Distribution of Categories
- Most Frequent Words

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
