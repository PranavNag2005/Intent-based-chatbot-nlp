# Intent-based-chatbot-nlp
Aim
The aim of this project is to develop an intents-based chatbot using Natural Language Processing (NLP) techniques and Logistic Regression. The chatbot will provide a user-friendly interface to classify user inputs and deliver meaningful responses, enhancing the interactive experience.

Learning Objectives

The objectives of this project are to:

Learn how chatbots process user input, recognize intents, and generate responses.

Use tokenization and TF-IDF vectorization to preprocess and analyze textual data.

Train and evaluate a Logistic Regression model for intent classification.

Create and deploy an interactive chatbot interface using the Streamlit framework.

About the Project
This project focuses on developing a functional chatbot capable of understanding user inputs by identifying intents and extracting entities. By leveraging NLP techniques and a Logistic Regression model, the chatbot interprets text inputs and provides appropriate responses. A Streamlit-based interface ensures user-friendly interaction, enabling seamless communication with the chatbot. This project serves as a foundational step toward creating advanced conversational agents, with scope for improvement through deeper datasets and sophisticated NLP techniques.

Data Source Link
The dataset used in this project consists of a labelled collection of intents, patterns, and responses stored in a JSON file.

Link: DATASET

Tools Used
Python:

Used as the core programming language for implementing the chatbot, preprocessing data, training the model, and deploying the application.

NLTK: Used for tokenization and preprocessing user input to make it suitable for training and predictions.

Scikit-learn:

TF-IDF Vectorizer: For transforming textual data into numerical representations suitable for machine learning.

Logistic Regression: Used as the machine learning algorithm for classifying intents based on user input.

Streamlit: Leveraged for developing an interactive chatbot interface where users can input text and view responses in real-time.

Jupyter Notebooks:

Employed Jupyter Notebooks for an interactive and collaborative coding environment. Jupyter Notebooks provided a seamless platform for code execution, visualization, and documentation.

Installation
Follow these steps to set up the project:

Clone the repository:


git clone https://github.com/PranavNag2005/Intent-based-chatbot-nlp.git
cd Intent-based-chatbot-nlp
Install the required dependencies:

bash
pip install -r requirements.txt
Download NLTK data:

python
import nltk
nltk.download('punkt')
Run the Streamlit application:


streamlit run app.py
