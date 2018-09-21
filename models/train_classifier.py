import sys
import sqlalchemy
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import grid_search
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
nltk.download('punkt')
import pickle


def load_data(database_filepath):
    """
    Load data from database into X and Y
    Args:
        database_filepath: path to database
    Returns:
        (DataFrame) X: feature (message)
        (DataFrame) Y: labels
    """

    # load data from database
    engine = sqlalchemy.create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('categorized_messages', engine)

    # Create X and Y
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1).values
    category_names = (df.iloc[:,4:].columns).tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text
    Args:
        text: Text to be tokenized
    Returns:
        (str[]): array of tokens
    """
    tokens = []
    stop = stopwords.words('english') + list(string.punctuation)

    for i in word_tokenize(text.lower()):
        if i not in stop:
            tokens.append(i)

    return tokens


def build_model():
    """
    Build MultiClassification Model
    Returns:
        (GridSearchCV) cv: GridSearchCV object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__learning_rate': [0.001, 0.01, 0.1, 1],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model perfomance
    Args:
        model: Classification model
        X_test: Test Set features
        Y_test: Test Set labels
        category_names: list of category names
    Returns:
        (str[]): array of tokens
    """

    Y_pred = model.predict(X_test)

    # Show results
    print(classification_report(Y_pred, Y_test, target_names=category_names))

    print("**** Accuracy scores for each category *****\n")
    for i in range(36):
        print("Accuracy score for " + category_names[i], accuracy_score(Y_test[:,i],Y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Save the model to a pickle file
    Args:
        model: Classification model
        model_filepath: Pickle file path
    """

    pickle.dump(model, open(model_filepath, "wb"))


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
