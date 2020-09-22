"""
Main for analysis of results.
"""

import matplotlib.pyplot as plt
import pandas
from tensorflow.keras.utils import plot_model

import classification
import datasets
import model_use
import preprocessing


def plot_categorized_pie(predictions):
    categories = classification.labels()

    labels = [classification.predication_to_label(prediction[0])
              for prediction in predictions]
    category_count = [labels.count(category) for category in categories]

    plt.pie(category_count, labels=categories, autopct='%1.1f%%')



print('Loading dataset') #load data
data_texts, data_labels = datasets.load_test_data()
print('Loading model') #load model
model, tokenizer = model_use.load_model()

print('Preprocessing on dataset')
data_inputs, _, _ = preprocessing.preprocess(data_texts, tokenizer=tokenizer)

print('Running predictions')
predictions = model.predict(data_inputs)

print('Processing results')
plot_categorized_pie(predictions)


print('Showing graph')
plt.show()
