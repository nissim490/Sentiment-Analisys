"""
Main for test the model.
Loads trained model, and runs testing on it using the testing
dataset. Uses the validation method discussed in the article.
"""

import numpy as np

import datasets
import model_use
import preprocessing
import training

print('Loading dataset')
data_texts, data_labels = datasets.load_test_data()
print('Loading model')
model, tokenizer = model_use.load_model()


print('Preprocessing on dataset')
data_inputs, _, _ = preprocessing.preprocess(data_texts, tokenizer=tokenizer)

# Remove neutral data and labels from the csv file, since this is a binary classification model
result = np.where(data_labels == 2)[0]
data_inputs = np.delete(data_inputs, result, axis=0)
data_labels = np.delete(data_labels, result, axis=0)

print('Running validation')
#K=2 in k fold cross validation. n_splits=2
results = training.validate(model,
                            data_inputs, data_labels,
                            num_epoch=20,
                            n_splits=2)
print (results)

accuracies = [result[1][1] for result in results]

print('Fold Accuracies:', ', '.join([str(a) for a in accuracies]))
print('Average Accuracy:', sum(accuracies) / len(accuracies))
