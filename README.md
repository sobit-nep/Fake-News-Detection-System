# Fake News Detection System

## Overview

This repository contains the source code for a Fake News Detection System developed using Natural Language Processing (NLP) techniques. The model is built using GloVe embeddings and a Long Short-Term Memory (LSTM) neural network. Additionally, the project is integrated with Flask to provide a web-based user interface for easy interaction.

## Model Architecture

The neural network model is defined as follows:

```python
model = Sequential()
model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25))
model.add(LSTM(units=64, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
```
![image](https://github.com/sobit-nep/Fake-News-Detection-System/assets/65544518/4147dcca-e90e-4707-8c89-37c7bbecc61c)


## Model Training
![image](https://github.com/sobit-nep/Fake-News-Detection-System/assets/65544518/3f2dc579-7d2d-4749-b389-e67cb0ccad2a)

## Model Evaluation
![image](https://github.com/sobit-nep/Fake-News-Detection-System/assets/65544518/e9b1f82d-74cd-44ca-8020-42eb6b660238)
![image](https://github.com/sobit-nep/Fake-News-Detection-System/assets/65544518/9fac9e11-2ee3-47cb-bc5c-80a7660b85c0)
![image](https://github.com/sobit-nep/Fake-News-Detection-System/assets/65544518/76bf4e4f-d1fd-4e92-ab2a-fdbd0c246edb)
![image](https://github.com/sobit-nep/Fake-News-Detection-System/assets/65544518/bf9327e4-d29c-413f-ad6d-00a9d121f0ea)



## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sobit-nep/fake-news-detection.git
    cd fake-news-detection
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the GloVe embeddings file and place it in the `models/` directory.

4. Download the trained model file (`final_model.h5`), the embedding matrix file (`embedding_matrix.npy`), and the tokenizer file (`tokenizer.pkl`) and place them in the `models/` directory.

## Usage

1. Run the Flask web application:

    ```bash
    python main.py
    ```

2. Open your web browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

3. Input the news title and text or provide a URL to a news article.

4. Click the "Submit" button to get the model's prediction on whether the news is fake or true.

## Preprocessing and Prediction

The `predict_fake_news` function in `main.py` handles the preprocessing and prediction:

- Input text is preprocessed to remove HTML tags, square brackets, and stopwords.
- Tokenization and padding are applied using the saved tokenizer.
- The model prediction is made, and confidence thresholds are used to classify the news.

## UI
![image](https://github.com/sobit-nep/Fake-News-Detection-System/assets/65544518/6657be17-79d9-4144-8d45-55230ae7485f)
![image](https://github.com/sobit-nep/Fake-News-Detection-System/assets/65544518/ba1cfd9b-bf51-45ce-bb9f-a44c5db88383)


## Acknowledgements

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Flask](https://flask.palletsprojects.com/)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)
- [TensorFlow](https://www.tensorflow.org/)


---

Feel free to customize the README.md to include additional sections such as project structure, future improvements, and any other relevant information.
