# Made by @yyoavv

import pandas as pd
import spacy
from spacy.util import minibatch
import random

def load_data(csv_file, split=0.9):
# Functionallity: Load data
# Input: csv_file - data source, split - the ratio of split the data between training and validation.
# Output: return losses - dictionary with loss function's results.

    data = pd.read_csv(csv_file)

    # Shuffle data
    train_data = data.sample(frac=1, random_state=7)

    texts = train_data.text.values
    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)}
              for y in train_data.sentiment.values]
    split = int(len(train_data) * split)

    train_labels = [{"cats": labels} for labels in labels[:split]]
    val_labels = [{"cats": labels} for labels in labels[split:]]

    return texts[:split], train_labels, texts[split:], val_labels


def train(model, train_data, optimizer):
# Functionallity: Train our model
# Input: model - model to train, train data - training dataset, optimizer.
# Output: return losses - dictionary with loss function's results.
    losses = {}
    random.seed(1)
    random.shuffle(train_data)

    batches = minibatch(train_data, size=8)
    for batch in batches:
        # train_data is a list of tuples
        # Split batch into texts and labels
        texts, labels = zip(*batch)

        # Update model with texts and labels
        model.update(texts, labels, sgd=optimizer, losses=losses)

    return losses


def predict(model, texts):
# Functionallity: Predict negativity ot positivity level of text.
# Input: model - model to predict, texts - texts to predict.
# Output: return predicted_class - dictionary with loss function's results.

    # Use the model's tokenizer to tokenize each input text
    docs = [model.tokenizer(text) for text in texts]

    # Use textcat to get the scores for each doc (scores is numpy array).
    scores, _ = model.get_pipe('textcat').predict(docs)

    # From the scores of negative and positive, find the class with the highest score/probability, list of 1 or 0.
    predicted_class = scores.argmax(axis=1)

    return predicted_class


def evaluate(model, texts, labels):
# Functionallity: Evaluate the accuracy of the model.
# Input: model - model to evaluate, texts - texts samples, labels - true labels of texts.
# Output: return predicted_class - dictionary with loss function's results.

    # Get predictions from textcat model (using your predict method)
    predicted_class = predict(model,texts)

    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = [int(each['cats']['POSITIVE']) for each in labels]

    # A boolean or int array indicating correct predictions
    correct_predictions = predicted_class == true_class

    # The accuracy, number of correct predictions divided by all predictions
    accuracy = correct_predictions.mean()
    return accuracy
