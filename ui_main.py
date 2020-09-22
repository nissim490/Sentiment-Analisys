"""
Main for UI prediction with the model.
Loads trained model, and runs UI allowing to see prediction results
on a tweet input.
"""

import classification
import model_use
import preprocessing
import ui


def create_run_model(model, tokenizer, sentence_padding=2000):
    def run(text):
        text, _, _ = preprocessing.preprocess([text],
                                              tokenizer=tokenizer,
                                              padding_size=sentence_padding)
        predication = model.predict(text)
        result = predication[0][0]
        return result, classification.predication_to_label(result)

    return run


print('Load model')
model, tokenizer = model_use.load_model() #load the files in Save folder that we save earlier
print('Done loading model')

print('Launching UI')
ui.show(create_run_model(model, tokenizer))
