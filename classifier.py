# Made by @yyoavv

import spacy
from spacy.util import minibatch
import random
import helpers

# Load data
train_texts, train_labels, val_texts, val_labels = helpers.load_data('yelp_ratings.csv')

# Create an empty model
nlp = spacy.blank("en")

# Create the TextCategorizer with exclusive classes and "bag of words" architecture, textcat - name of pipe
textcat = nlp.create_pipe(
              "textcat",
              config={
                "exclusive_classes": True,
                "architecture": "bow"})

# Add the TextCategorizer to the empty model
nlp.add_pipe(textcat)

# Add labels to text classifier
textcat.add_label("NEGATIVE")
textcat.add_label("POSITIVE")

# Training
optimizer = nlp.begin_training()
train_data = list(zip(train_texts, train_labels))

n_iters = 5
for i in range(n_iters):
    losses = helpers.train(nlp, train_data, optimizer)
    accuracy = helpers.evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")

# Prediction
texts = val_texts[34:38]
predictions = helpers.predict(nlp, texts)

# Evaluation
for p, t in zip(predictions, texts):
    print(f"{textcat.labels[p]}: {t} \n")
