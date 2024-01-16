# Basic libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import gensim
import numpy as np
import nltk
import random
import itertools
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# Preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from itertools import combinations
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
#from imblearn.under_sampling import NearMiss, RandomUnderSampler
#from imblearn.over_sampling import SMOTE, ADASYN
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Models

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
#import lightgbm as lgb

# Evaluation

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, make_scorer
#from lime import lime_text
#from sklearn.pipeline import make_pipeline
#from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings("ignore")
import requests
from annotated_text import annotated_text

def query_plain(text, url="http://bern2.korea.ac.kr/plain"):
    response_data = requests.post(url, json={'text': text}).json()

    # Extract annotations and text from response data
    annotations = response_data["annotations"]
    text = response_data["text"]

    # Sort annotations by 'begin' index
    annotations.sort(key=lambda annotation: annotation["span"]["begin"])

    # Initialize variables
    formatted_text = []
    last_index = 0

    # Iterate through each annotation
    for annotation in annotations:
        begin, end = annotation["span"]["begin"], annotation["span"]["end"]

        # Add the text before the current annotation
        if begin > last_index:
            formatted_text.append(text[last_index:begin])

        # Add the annotated text
        mention = text[begin:end]
        obj = annotation["obj"]
        formatted_text.append((mention, obj))

        last_index = end

    # Add any remaining text after the last annotation
    if last_index < len(text):
        formatted_text.append(text[last_index:])
    print(formatted_text)
    # Display the formatted text with annotated_text
    annotated_text(formatted_text)



st.title("Medical Text Classifier 🩺👨‍⚕️")
# 2. horizontal menu
selected2 = option_menu(None, ["Home", "EDA", "Train", 'About'],
    icons=['house', 'data-icon', "computer", 'about'],
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected2 == "Home":
    text = st.text_area("Input your medical transcription here 📋", value="", height=500)
    if st.button("Analyse 🔎", type="primary"):
        query_plain(text)

elif selected2 == "EDA":
    data = pd.read_csv('mtsamples.csv')
    data = data.iloc[:, 1:]
    st.write("Number of Samples:", data.shape[0])
    st.write("Example of the first lines of the dataset")
    st.dataframe(data.head())

    # Data Preprocessing
    data = data[['transcription', 'medical_specialty']]
    data = data.drop(data[data['transcription'].isna()].index)
    st.write("Sample Transcription:\n", data.iloc[4]['transcription'])

    # Medical Specialty Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    specialty_counts = data['medical_specialty'].value_counts()
    sns.barplot(x=specialty_counts.values, y=specialty_counts.index, ax=ax)
    ax.set_title('Medical Specialty Distribution')
    ax.set_ylabel('Specialty')
    ax.set_xlabel('Count')
    st.pyplot(fig)

    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    data["tokens"] = data["transcription"].apply(tokenizer.tokenize)

    all_words = [word for tokens in data["tokens"] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in data["tokens"]]
    VOCAB = sorted(list(set(all_words)))
    st.write(f"{len(all_words)} words total, with a vocabulary size of {len(VOCAB)}")
    st.write(f"Max sentence length is {max(sentence_lengths)}")

    # Sentence Length Histogram
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(sentence_lengths, edgecolor='black', bins=20)
    ax.set_title('Sentence Length Histogram')
    ax.set_xlabel('Sentence length')
    ax.set_ylabel('Number of sentences')
    st.pyplot(fig)

    st.write(f"Median sentence length: {np.median(sentence_lengths)}")
    st.write(f"Mean sentence length: {round(np.mean(sentence_lengths), 2)}")

    # Word Cloud
    all_transcriptions = " ".join(data["transcription"])
    wordcloud = WordCloud(width=1200, height=800, background_color='white', min_font_size=10).generate(
        all_transcriptions)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud)
    ax.axis("off")
    ax.set_title("Word Cloud")
    st.pyplot(fig)

elif selected2 == "Train":
    data = pd.read_csv('mtsamples.csv')
    data = data.iloc[:, 1:]

    # Data Preprocessing
    data = data[['transcription', 'medical_specialty']]
    data = data.dropna(subset=['transcription'])
    data = data.dropna(subset=['medical_specialty'])

    st.header("Training Steps:")
    st.write("1 - Pre-processing, Cleaning, Tokenization and Lemmatization, Stopwords Removal")
    st.write("2 - Feature Extraction")
    option = st.selectbox(
        'Select Feature Extraction Method',
        ('BOW', 'TF-IDF (1n-gram)', 'TF-IDF (2n-gram)', "Word2Vec", "BioWord2Vec"))
    st.write("3 - Algorithm")
    option2 = st.selectbox(
        'Select The Dataset Balancing Method',
        ('None', 'SMOTE', 'ADASYN', "Undersample", "Cut Target Classes"))
    st.write("4 - Algorithm")
    option3 = st.selectbox(
        'Select Feature Extraction Method',
        ('BOW', 'TF-IDF', 'Mobile phone'))
    if st.button("Train ⚙️🤖", type="primary"):

        if option2 == "None":
            st.write("No balancing was applied")
        elif option2 == "SMOTE":
            smote = SMOTE()
            data['transcription'], data['medical_specialty'] = smote.fit_resample(data[['transcription']],
                                                                                  data['medical_specialty'])
            st.write("SMOTE balancing applied")
        elif option2 == "Undersample":
            # Find the number of samples in the smallest class
            min_count = data['medical_specialty'].value_counts().min()
            data = data.groupby('medical_specialty').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
            st.write("Undersampling applied")
        elif option2 == "Cut Target Classes":
            # Find the count of the majority class
            max_count = data['medical_specialty'].value_counts().max()
            # Define threshold as 20% of the majority class count
            threshold = 0.2 * max_count
            # Filter classes that meet the threshold
            valid_classes = data['medical_specialty'].value_counts()[
                data['medical_specialty'].value_counts() > threshold].index
            data = data[data['medical_specialty'].isin(valid_classes)]
            st.write("Cut Target Classes applied")



        # Text preparation


        #Load English tokenizer, tagger, parser, NER, and word vectors
        nlp = spacy.load("en_core_web_sm")


        @st.cache_data
        def basic_preprocessing(df):
            # Assuming df is your DataFrame
            # Step 1: Lowercase Conversion

            df['transcription'] = df['transcription'].fillna('')

            df['transcription'] = df['transcription'].str.lower()

            # Step 2: Removing Punctuation and Numbers
            df['transcription'] = df['transcription'].apply(lambda x: re.sub(r'[\d\W]+', ' ', x))

            # Step 3: Tokenization
            df['transcription'] = df['transcription'].apply(nltk.word_tokenize)

            # Step 4: Lemmatization
            lemmatizer = WordNetLemmatizer()
            df['transcription'] = df['transcription'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

            # Step 5: Removing Stop Words
            stop_words = set(stopwords.words('english'))
            df['transcription'] = df['transcription'].apply(lambda x: [word for word in x if word not in stop_words])

            # Step 6: Encode 'medical_speciality' Column
            # One-hot encoding 'medical_speciality' column
            one_hot_encoder = OneHotEncoder(sparse=False)
            medical_specialty_encoded = one_hot_encoder.fit_transform(df[['medical_specialty']])

            # Convert the encoded result into a DataFrame
            encoded_df = pd.DataFrame(medical_specialty_encoded,
                                      columns=one_hot_encoder.get_feature_names_out(['medical_specialty']))

            # Concatenate the original DataFrame with the new one-hot encoded columns
            df = pd.concat([df, encoded_df], axis=1)

            # Optionally, you can drop the original 'medical_speciality' column
            # df.drop('medical_specialty', axis=1, inplace=True)

            df = df.dropna(subset=['transcription'])

            return df


        from string import punctuation
        def preprocess_for_biosentvec(text):
            # Basic cleaning
            text = text.lower()
            text = re.sub(r'[\d]', ' ', text)  # Remove digits
            text = re.sub(r'[/]', ' / ', text)  # Separate slashes
            text = re.sub(r'[^\w\s]', ' ', text)  # Remove other non-word characters

            # Tokenization and rejoining to form a cleaned sentence
            tokens = [token for token in word_tokenize(text) if token not in punctuation]
            return ' '.join(tokens)

        # Introduce evaluation metrics
        def get_metrics(y_test, y_predicted):
           precision = precision_score(y_test, y_predicted, average='weighted')

           recall = recall_score(y_test, y_predicted, average='weighted')

           f1 = f1_score(y_test, y_predicted, average='weighted')

           accuracy = accuracy_score(y_test, y_predicted)
           return accuracy, precision, recall, f1

        df_temp = data.copy(deep=True)
        df_temp2 = data.copy(deep=True)

        ########################Feature Extraction########################################

        from sklearn.feature_extraction.text import CountVectorizer


        def extract_features_bow(df, ngram_range=(1, 1)):
            vectorizer = CountVectorizer(ngram_range=ngram_range)
            features = vectorizer.fit_transform(df['transcription'].astype(str))
            return features, vectorizer.get_feature_names_out()

        from sklearn.feature_extraction.text import TfidfVectorizer
        def extract_features_tfidf(df, ngram_range=(1, 1)):
            vectorizer = TfidfVectorizer(ngram_range=ngram_range)
            features = vectorizer.fit_transform(df['transcription'].astype(str))
            return features, vectorizer.get_feature_names_out()

        def extract_features_w2v(df):
            # Word2Vec expects a list of lists of tokens
            df['transcription'] = df['transcription'].fillna('')
            sentences = df['transcription'].tolist()


            # Train the Word2Vec model
            model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

            # Function to average word vectors for a text
            def document_vector(word_list):
                # Remove out-of-vocabulary words and get their vectors
                doc = [word for word in word_list if word in model.wv.key_to_index]
                return np.mean(model.wv[doc], axis=0) if len(doc) > 0 else np.zeros(model.vector_size)

            # Apply the function to each row of the DataFrame
            doc_vectors = np.vstack(df['transcription'].apply(document_vector))

            return doc_vectors, model


        #######BIO WORD2VEC##############

        def document_vector_biov(text, model):
            # Tokenize the text and get embeddings for each word
            words = word_tokenize(text)
            word_vectors = [model[word] for word in words if word in model.key_to_index]

            # Aggregate the word vectors to get a single document vector
            return np.mean(word_vectors, axis=0) if len(word_vectors) > 0 else np.zeros(model.vector_size)


        def extract_features_biov(df, model):
            # Preprocess and extract features for each transcription
            df['preprocessed'] = df['transcription'].apply(preprocess_for_biosentvec)
            feature_matrix = np.vstack(df['preprocessed'].apply(lambda x: document_vector_biov(x, model)))

            return feature_matrix

        #########APLY##########

        features = None
        if option == "BOW":
            with st.spinner('Pre-Processing and Applying BOW'):
                df_temp = basic_preprocessing(df_temp)
                features = extract_features_bow(df_temp, ngram_range=(1, 2))
        elif option == "TF-IDF (1n-gram)":
            with st.spinner('Pre-Processing and Applying TF-IDF (1n-gram)...'):
                df_temp = basic_preprocessing(df_temp)
                features, tfidf_vocab = extract_features_tfidf(df_temp, ngram_range=(1, 1))
        elif option == "TF-IDF (2n-gram)":
            with st.spinner('Pre-Processing and Applying TF-IDF (2n-gram)...'):
                df_temp = basic_preprocessing(df_temp)
                features, tfidf_vocab = extract_features_tfidf(df_temp, ngram_range=(1, 2))
        elif option == "Word2Vec":
            with st.spinner('Pre-Processing and Applying Word2Vec...'):
                df_temp = basic_preprocessing(df_temp)
                st.dataframe(df_temp)
                features, w2v_model = extract_features_w2v(df_temp)
        elif option == "BioWord2Vec":
            with st.spinner('Pre-Processing and Applying BioWord2Vec...'):
                filename = 'BioWordVec.bin'
                model = KeyedVectors.load_word2vec_format(filename, binary=True)
                features = extract_features_biov(df_temp2, model)

        print("HEÇLO")
        print(features)

elif selected2 == "About":
    st.write("Project created for Faculdade de Engenharia do Porto (FEUP)"
            " Natural Language Processing Course - ProDEI 2023/2024")
    st.write("Tiago Fonseca - up202302320")

