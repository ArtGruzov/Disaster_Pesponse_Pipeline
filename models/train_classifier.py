import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import sys
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Table', 'sqlite:///' + database_filepath)
    X = df.message
    Y = df.drop(labels=['id', 'message', 'original', 'genre'], axis=1)
    return X, Y


def tokenize(text):
    return word_tokenize(text)


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 5000)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test):
    best_estimator = model.best_estimator_
    model_result = best_estimator.predict(X_test)

    categories = list(Y_test.columns)
    model_result = pd.DataFrame(model_result)
    Y_test = pd.DataFrame(Y_test.values)

    for i in range(model_result.shape[1]):
        print('PREDICTION RESULT: {}\n\n'.format(categories[i]), classification_report(Y_test[i], model_result[i]))
        print('==='*25, '\n')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
