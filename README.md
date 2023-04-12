# Sentiment analysis with deep-learning approach

This project allows you to get sentiment for the entered sentence or text.
Sentiments are emotions, among them approval, curiosity, neutral, nervousness, surprise, sadness, fear.
The main file is main.py and can be run from the command line. 
For correct operation, the mandatory parameter --sentence is required (the text for which sentiments will be issued)

To get a prediction, you need to follow these steps:
* First initialize and run the virtual environment
  * Install the virtualenv package. You can install it with pip: `pip install virtualenv`
  * Create the virtual environment. For example to create one in the local directory called ‘mypython’, type the following: `virtualenv mypython`
  * Activate the virtual environment. Mac OS / Linux - `source mypython/bin/activate`, Windows - `mypthon\Scripts\activate`
* Download the requirements for the project from the requirements.txt file with the command `pip install -r requirements.txt`
* Go to the root folder of the project and run the command: `python main.py --sentence="your sentence"`

After executing the command, you will receive the result of the model described in the format 

`Your input text: "your sentence" Predicted emojis: "predicted sentiments"`
