import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import pickle
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
     clean ,tokenize and lemmatize text 
    """
    text = re.sub(r'\W',' ',text)
    token = word_tokenize(text)
    lemm = [WordNetLemmatizer().lemmatize(i.lower(),pos='v') for i in token]
    stem = [PorterStemmer().stem(i) for i in lemm]
    return stem

# load data
engine = create_engine('sqlite:///../data/disaster_messages.db')
df = pd.read_sql_table('message categories', engine)

# load model
file_ = open("../models/classifier.pkl",'rb')
model = pickle.load(file_)
    


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category = df.iloc[:,4:]
    categories_names = list(category.columns)
    categories_occurrencies = category.sum().sort_values(ascending=False)

    genre_water = df.groupby('genre')['water'].count()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x= categories_names,
                    y=categories_occurrencies
                )
            ],

            'layout' : {
                'title' : 'Number of each category occurrencies',
                'yaxis': {
                    'title':"count of each category"
                },
                'xaxis':{
                    'title':"categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x= genre_names,
                    y=genre_water
                )
            ],

            'layout' : {
                'title' : 'water related to which genre',
                'yaxis': {
                    'title':"count of genres related to water"
                },
                'xaxis':{
                    'title':"genres"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()