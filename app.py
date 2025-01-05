import os
import nltk
import ssl
import json
import streamlit as st
import random
import datetime
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set the page title and favicon
st.set_page_config(
    page_title="BotAura: Enlightened Conversations",  # Change this to your desired title
    page_icon="🤖",  # You can use an emoji or a URL to an image file
)

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=1000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Function to check file size and create a new file if needed
def check_and_create_new_file(file_path, max_size_mb=5):
    if os.path.exists(file_path):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            base, ext = os.path.splitext(file_path)
            new_file_path = f"{base}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
            return new_file_path
    return file_path

def submit_input():
    if 'last_submit_time' in st.session_state:
        now = datetime.datetime.now()
        diff = now - st.session_state.last_submit_time
        if diff.total_seconds() < 1:  # Adjust the debounce interval as needed
            return
    st.session_state.last_submit_time = datetime.datetime.now()
    
    user_input = st.session_state.input_text
    if user_input.strip() != "":  # Ensure input is not empty or whitespace
        response = chatbot(user_input)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state.messages.append((user_input, response))
        st.session_state.input_text = ""  # Clear the input_text after processing

        # Check file size and get the new file path if needed
        file_path = check_and_create_new_file('chat_log.csv')
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([user_input, response, timestamp])

def handle_button_press():
    submit_input()
    st.session_state.input_text = ""  # Clear the input_text after processing

def display_message(user_message, bot_message):
    st.markdown(f'<div class="chat-message user-message">{user_message}</div>', unsafe_allow_html=True)
    if "```" in bot_message:
        st.code(bot_message.replace("```", "").strip())  # Use st.code for code blocks and remove backticks
    else:
        st.markdown(f'<div class="chat-message bot-message">{bot_message}</div>', unsafe_allow_html=True)

def main():
    st.markdown(
        """<style>
        .stApp { background-color: #f8f9fa; }
        .sidebar .sidebar-content .element-container .element-label { color: #93a1a1; font-size: 20px; font-weight: bold; cursor: pointer; }
        .sidebar .sidebar-content .element-container .element-radio { background-color: #073642; }
        .sidebar .sidebar-content .element-container:hover .element-label { cursor: pointer; }
        .chat-container { height: 60vh; overflow-y: auto; background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 50px; }
        .chat-message { margin-bottom: 15px; padding: 15px; border-radius: 10px; max-width: 70%; word-wrap: break-word; width: 90%; } /* Adjusted width */
        .user-message { background-color: #e8f5e9; width: 90%; } /* Adjusted width */
        .bot-message { background-color: #ffecb3; width: 90%; } /* Adjusted width */
        .small-text { font-size: 12px; color: #268bd2; text-align: center; margin-bottom: 50px; }
        .input-container { width: 100%; background-color: #ffffff; padding: 10px; border-top: 2px solid #0288d1; display: flex; align-items: center; position: fixed; bottom: 0; left: 0; }
        .input-container input { flex: 1; height: 50px; padding: 10px; border-radius: 5px; border: 1px solid #0288d1; transition: border-color 0.3s ease; width: 90%; } /* Adjusted width */
        .input-container input:focus { border-color: #0288d1; box-shadow: none; outline: none; }
        .input-container button { padding: 10px 15px; border: none; border-radius: 5px; background-color: #0288d1; color: white; cursor: pointer; margin-left: 10px; }
        .centered-title { text-align: center; color: #0288d1; margin-top: 20px; }
        .spaced-subheader { margin-top: 30px; }
        .search-input { width: calc(100% - 50px); }
        </style>""",
        unsafe_allow_html=True
    )

    menu = ["Chat Room", "History Records", "Bot Info"]
    choice = st.sidebar.selectbox("Chat Options", menu, format_func=lambda x: f"❱ {x}")

    if choice == "Chat Room":
        st.markdown('<h1 class="centered-title">Interactive Chat Room</h1>', unsafe_allow_html=True)
        st.write("Hello and welcome to our chatbot! 🤖 Type a message and press enter to begin. I'm here to assist you with any questions or information you need.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'input_text' not in st.session_state:
            st.session_state.input_text = ''
        
        if 'last_submit_time' not in st.session_state:
            st.session_state.last_submit_time = datetime.datetime.now()

        # Chat container
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            for message in st.session_state.messages:
                user_message, bot_message = message
                display_message(user_message, bot_message)

        # Input container
        with st.container():
            user_input_col, button_col = st.columns([8, 1])
            with user_input_col:
                st.text_input("Type your message here...", key="input_text", placeholder="Type your message here...", on_change=submit_input, label_visibility="collapsed")
            with button_col:
                st.button("⬆️", on_click=handle_button_press)

        st.markdown('<p class="small-text">Chatbot will make mistakes. Double-check the response.</p>', unsafe_allow_html=True)

    elif choice == "History Records":
        st.markdown('<h1 class="centered-title">Chat History Records</h1>', unsafe_allow_html=True)
        st.subheader("History Records", anchor=False)
        search_query = st.text_input("Search Chat History", key="search_query", placeholder="Search messages...")

        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            if csv_reader:
                next(csv_reader)  # Skip the header row
            for row in reversed(list(csv_reader)):
                if len(row) >= 3 and (not search_query or search_query.lower() in row[0].lower() or search_query.lower() in row[1].lower()):
                    st.markdown(f"**Timestamp:** {row[2]}")
                    st.markdown(f"**User Input:** {row[0]}")
                    st.markdown(f"**Chatbot Response:** {row[1]}")
                    st.markdown("---")

    elif choice == "Bot Info":
        st.header("Chatbot Implementation Using NLP and Logistic Regression")

        st.subheader("Project Overview")
        st.write("""
        This project aims to create a sophisticated chatbot capable of understanding and responding to user inputs through Natural Language Processing (NLP) techniques. By leveraging Logistic Regression for intent classification, the chatbot offers a seamless and interactive user experience. Below is a breakdown of the key components and objectives of the project.
        """)

        st.subheader("Problem Statement")
        st.write("The goal is to develop a chatbot that can accurately identify user intents and provide relevant responses, thus facilitating smooth and efficient interactions.")

        st.subheader("Aim")
        st.write("To develop an intents-based chatbot utilizing NLP techniques and Logistic Regression, ensuring a user-friendly interface that classifies user inputs and delivers meaningful responses.")

        st.subheader("Learning Objectives")
        st.write("""
        - Understand chatbot mechanisms for processing user input and recognizing intents.
        - Explore tokenization and TF-IDF vectorization for text preprocessing.
        - Train and evaluate a Logistic Regression model for intent classification.
        - Develop and deploy an interactive chatbot interface using Streamlit.
        """)

        st.subheader("About the Project")
        st.write("""
        The chatbot is designed to comprehend user inputs by identifying intents and extracting relevant entities. Using NLP techniques and a Logistic Regression model, it interprets text inputs and provides appropriate responses. The Streamlit-based interface ensures a user-friendly interaction, making communication with the chatbot smooth and intuitive.

        Key Highlights:
        - **Intuitive Design:** The chatbot is designed to be easy to use and responsive, making it suitable for a wide range of applications, from customer support to virtual assistants.
        - **Training and Accuracy:** Trained on a labeled dataset of intents and patterns, the chatbot is capable of managing a variety of queries with high accuracy and efficiency.
        - **Seamless Interaction:** The Streamlit interface ensures that users can interact with the chatbot in a straightforward and engaging manner.
        """)

        st.subheader("Data Source")
        st.write("The dataset consists of labeled intents, patterns, and responses stored in a JSON file, serving as the training data for the chatbot.")
        st.markdown("[Link to the Dataset](https://github.com/PranavNag2005/Intent-based-chatbot-nlp/tree/main)")

        st.subheader("Tools and Techniques")
        st.write("""
        - **Python:** The primary programming language used for implementation, data preprocessing, model training, and deployment.
        - **NLTK (Natural Language Toolkit):** Utilized for tokenizing and preprocessing user input, converting text into a format suitable for machine learning models.
        - **Scikit-learn:** 
            - **TF-IDF Vectorizer:** Converts textual data into numerical representations, enabling the model to process and analyze text.
            - **Logistic Regression:** Employed to classify user intents based on processed text data.
        - **Streamlit:** Used to create an interactive web interface, allowing users to interact seamlessly with the chatbot.
        """)

        st.subheader("Future Enhancements")
        st.write("""
        - **Expanded Dataset:** Incorporating a more diverse set of intents and queries to improve the chatbot's comprehensiveness.
        - **Advanced NLP Techniques:** Implementing deep learning models such as LSTM or BERT to enhance the chatbot's accuracy and performance.
        - **Contextual Conversations:** Introducing features like context retention and multi-turn conversations to make interactions more natural and human-like.
        - **Voice Integration:** Adding voice recognition and response capabilities to provide a more dynamic user experience.
        - **Multi-language Support:** Expanding the chatbot's language capabilities to cater to a global audience.
        """)


if __name__ == "__main__":
    main()
