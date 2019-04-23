# Disaster Response Pipeline Project

### Project Overview
There's a data set containing real messages that were sent during disaster events in this project. I created a machine 
learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Description of the files 

<b>app/</b> <br>
- <i>run.py:</i> flask file for running the app

<b>app/template/</b> <br>
- <i>master.html:</i> the main page of web app
- <i>go.html:</i> another page of web app

<b>data/</b> <br>
- <i>disaster_categories.csv:</i> entry categorical data
- <i>disaster_messages.csv:</i> entry messages
- <i>process_data.py:</i> script to process the data
- <i>InsertDatabaseName.db:</i> database to save processed data

<b>model/</b> <br>
- <i>train_classifier.py:</i> script to build the model
- <i>classifier.pkl:</i> the builded model

### Files structure
```html
- app
| - template 
| |- master.html 
| |- go.html
|- run.py 

- data
|- disaster_categories.csv 
|- disaster_messages.csv 
|- process_data.py 
|- InsertDatabaseName.db 

- models
|- train_classifier.py
|- classifier.pkl 

- README.md
```
### Screenshots
![Снимок экрана 2019-04-22 в 22 08 10](https://user-images.githubusercontent.com/40493554/56552966-8e184b00-6595-11e9-83b8-b34b8fcd6b6a.png)
![Снимок экрана 2019-04-22 в 22 07 36](https://user-images.githubusercontent.com/40493554/56552971-91133b80-6595-11e9-800b-0c8bdfb7ef16.png)
