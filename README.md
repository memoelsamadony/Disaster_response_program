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
