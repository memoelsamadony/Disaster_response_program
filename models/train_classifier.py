#import packages
import sys
import pandas as pd
import numpy as np 
from sqlalchemy import create_engine
import re
import pickle


from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as cr
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import warnings
warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

def load_data(database_filepath):
    """loads data from database
    Args:
    database_filepath : the path to the database to extract data from it
    
    Returns : 
    X : the messages column to be tokenized and cleaned
    y : the categories to be predicted
    categories names : the names of the categories to be predicted
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('message categories',engine)
    X = df['message']
    y = df.iloc[:,4:]
    categories = list(y.columns)

    return X,y,categories


def tokenize(text):
    """clean ,tokenize and lemmatize text
    Args:
    text: the message to be cleaned, tokenized ,lemmatized and stemmed

    Returns : cleaned list of words 
    """
    text = re.sub(r'\W',' ',text)
    token = word_tokenize(text)
    lemm = [WordNetLemmatizer().lemmatize(i.lower(),pos='v') for i in token]
    stem = [PorterStemmer().stem(i) for i in lemm]
    return stem




def build_model():
    pipeline1 = Pipeline([('vect',CountVectorizer(tokenizer = tokenize,stop_words='english')),
                     ('tfidf',TfidfTransformer()),
                     ('multi output classifier',MultiOutputClassifier(RidgeClassifier()))],verbose=True)
    
    return pipeline1


def evaluate_model(model, X_test, y_test, category_names):
    """Prints out classification report and Calculate average accuracy , precision , recall and f1 score
    Args:
    model : model used to be tested
    X_test : messages column for testing
    y_test : true values of predictions
    
    Returns : None
    prints out classification report , average accuracy , precision , recall and f1 score
    
    """
    y_preds = model.predict(X_test)
    y_preds = pd.DataFrame(y_preds, columns = y_test.columns)
    acc = []
    precision = []
    recall = []
    f1 = []
    for i  , j in zip(range(len(list(y_test.columns))),y_test.columns):
        print('category : ',j)
        print(cr(y_test.iloc[:,i],y_preds.iloc[:,i]))
        print('---------------------------')
        acc.append(accuracy_score(y_test.iloc[:,i],y_preds.iloc[:,i]))
        precision.append(precision_score(y_test.iloc[:,i],y_preds.iloc[:,i],average='macro'))
        recall.append(recall_score(y_test.iloc[:,i],y_preds.iloc[:,i],average='macro'))
        f1.append(f1_score(y_test.iloc[:,i],y_preds.iloc[:,i],average='macro'))
    
    print('average accuracy score : {:.2f}'.format(np.mean(acc)))
    print('average precision score : {:.2f}'.format(np.mean(precision)))
    print('average recall score : {:.2f}'.format(np.mean(recall)))
    print('average f1 score : {:.2f}'.format(np.mean(f1)))


def save_model(model, model_filepath):# saves the model to a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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