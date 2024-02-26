# Run this command first: pip install -r requirements.txt
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup

# Load your trained model
model = load_model('models/final_model.h5')
# Modify the embedding layer weights
embedding_matrix = np.load('models/embedding_matrix.npy')
model.get_layer('embedding').set_weights([embedding_matrix])

# Assuming df is your DataFrame
def predict_fake_news(input_text):
    # Define a set of stopwords
    stop = set(stopwords.words('english'))

    # Preprocess the input text
    def denoise_text(text):
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        text = remove_stopwords(text)
        return text

    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)

    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)

    input_text = denoise_text(input_text)
    max_features = 10000

    # Tokenize and pad the input text
    tokenizer = Tokenizer(num_words=max_features)
    with open('models/tokenizer.pkl', 'rb') as file:
        tokenizer_word_index = pickle.load(file)
    tokenizer.word_index = tokenizer_word_index

    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequences, maxlen=300)

    # Make a prediction using your model
    prediction = model.predict(padded_sequence)

    confidence_thresholds = {
        'false': 0.5,       # If prediction is between 10% and 50%
        'half_true': 0.75,  # If prediction is between 50% and 75%
        'mostly_true': 0.9  # If prediction is between 75% and 90%
    }
    prediction_value = prediction[0][0]

    if prediction_value < confidence_thresholds['false']:
        result = 'FAKE/UNKNOWN NEWS'
    elif prediction_value < confidence_thresholds['half_true']:
        result = 'TRUE NEWS'
    elif prediction_value < confidence_thresholds['mostly_true']:
        result = 'TRUE NEWS'
    else:
        result = 'TRUE NEWS'

    print(prediction_value)
    return result

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    scraped_title = ''
    scraped_text = ''

    if request.method == 'POST':
        title = request.form.get('title', '')  # Use request.form.get to handle optional fields
        text = request.form.get('text', '')
        url = request.form.get('url', '')

        if not title and not text and not url:
            # Handle the case where no input is provided
            return render_template('index.html', prediction='Please provide input.')

        if url:
            try:
                response = requests.get(url)
                html_content = response.content

                # Create a Beautiful Soup object
                soup = BeautifulSoup(html_content, 'html.parser')

                # Extract titles and text
                titles = soup.find_all('h1')
                texts = soup.find_all('p')

                # Store titles and texts in lists
                title_list = [title.text for title in titles]
                text_list = [text.text for text in texts]

                # Concatenate titles starting from index 1
                scraped_title = " ".join(title_list)

                # Concatenate all texts
                scraped_text = " ".join(text_list[1:])
                input_text = scraped_title + ' ' + scraped_text
                prediction = predict_fake_news(input_text)
            except Exception as e:
                print(f"Error while processing URL: {e}")

        input_text = title + ' ' + text + ' ' + scraped_title + ' ' + scraped_text
        prediction = predict_fake_news(input_text)
        
        return render_template('index.html', title=title, text=text, prediction=prediction, scraped_title=scraped_title, scraped_text=scraped_text,  url=url)

    return render_template('index.html', prediction='')

if __name__ == '__main__':
    app.run(debug=True)
