"""
Main for training the model.
"""

import model_use
import datasets
import preprocessing
import training

DATASET_LIMIT = None

print('Loading dataset')
data, labels = datasets.load_train_data()
print('Preprocessing dataset')
data, tokenizer, weak_data = preprocessing.preprocess(data,
                                                      in_between_cleaning=training.weak_data_matcher,
                                                      tokenizer=None) # calling parmater with function

print('Creating model')
model = model_use.create(tokenizer)

print('Training')
training.train(model,
               data,
               labels,
               weak_data,
               num_epoch=40,
               batch_size=512,
               data_limit=DATASET_LIMIT)

print('Saving Model')
model_use.save_model(model, tokenizer)
