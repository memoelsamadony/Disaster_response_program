# Disaster response program
This is a web application is made as a project part of Udacity Data Scientist Nanodegree program ,And its developed to predict to which category a message belongs in 
time of disasters .



## About 
This web application takes a message from the user and then predict if it between 36 different categories . And the data that the model trained on was provided by [FigureEight](https://f8federal.com/) 

Main project Files :

- The app folder : 
               - templates folder :
                   - master.html : main page HTML code
                   - go.html : The page that the classification results appears on
               - run.py : the python file at which the Backend of the program operates
- The data folder :
               - categories.csv : the file that contains data about the categories
               - messages.csv : the file that contain data about the messages and thier genre
               - process_data.py : python file contains code for merging the above two files into a single dataframe and then cleans ,Then finally export it to a DB
               - disaster_messages.db : DB file that contains the table that was exported from process_data.py
- The models folder :
               - train_classifier.py : python file contains code for importing data from disaster_messages.db and build and train the model , Then evaluate it and exporting it to a pickle file
               - classifier.pkl : pickle file that contains the trained model
- The Notebooks folder : Folder that contains explantory notebooks to build both the ETL pipeline and the ML pipeline . Also contains copies of csv files and db file

## How to run the web app 
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/disaster_messages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_messages.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Then go to http://0.0.0.0:3000/

## Libraries 
- flask
- sklearn
- re
- pandas
- numpy
- plotly
- sqlalchemy
- nltk

To install those packges you can use pip or conda install
example `pip install flask pandas` or `py -m pip install numpy`


## The web app 
the web app has two pages , That is the landing page :

![image](https://user-images.githubusercontent.com/91777656/192622916-5f7c2387-4780-4cac-a291-66a70486cbfd.png)

And this the page wher the classification results appear :

![image](https://user-images.githubusercontent.com/91777656/192625330-86292ea8-852b-4c28-9d0e-f685c479d414.png)


## credits 
this [article](https://www.sciencedirect.com/science/article/pii/S1877050919314152) helped me understand imbalanced data
