import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import spacy
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='NGRAM', help='model to run (NGRAM, CNN, or BERT)')
    parser.add_argument('--target', type=str, default='EMPATHY', help='target to predict (EMPATHY, POLARITY, or INTENSITY)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=50, help='training batch size; 50 by default')

    # paths to data
    parser.add_argument('--train_path', type=str, default="../data/trac2_CONVT_train.csv", help='model training data')
    parser.add_argument('--dev_path', type=str, default="../data/trac2_CONVT_dev.csv", help='model development data')
    parser.add_argument('--blind_test_path', type=str, default="../data/goldstandard_CONVT.csv", help='model validation data')

    args = parser.parse_args()
    return args

# data parsing
class EmotionExample: 
    """
    Represents a single piece of training or testing data
    """
    def __init__(self, tokens: list | None, transformer_tokens: dict[str, list[int]] | None, emotional_intensity: float, emotional_polarity: float, empathy: float):
        #id,article_id,conversation_id,turn_id,"speaker_id","text",person_id,person_id_1,person_id_2,Emotion,EmotionalPolarity,Empathy
        self.tokens = tokens
        self.transformer_tokens = transformer_tokens
        self.emotional_intensity = emotional_intensity
        self.emotional_polarity = emotional_polarity
        self.empathy = empathy

def parse_dataset(data_path: str, model: str, tokenizer):
    """
    Given a path to a csv file containing training or test data, converts that data to a list of 
    EmotionExmples and returns the resulting list 
    """

    # read data as pandas dataframe
    df = pd.read_csv(data_path)
    emotion_examples = []

    # iterate through dataframe and create EmotionExample object for each
    for _, row in df.iterrows(): 
        sentence = row["text"]
        tokens = None
        transformer_tokens = None
        if model == "BERT":
            transformer_tokens = tokenizer(sentence, return_tensors=None)
        else:
            doc = tokenizer(sentence)              # run spaCy on the sentence
            tokens = [token.text for token in doc]   # extract token strings

        emotional_intensity = float(row["Emotion"])
        emotional_polarity = float(row["EmotionalPolarity"])
        empathy = float(row["Empathy"])
        # other row are just ids, probably won't need them

        emotion_item = EmotionExample(tokens, transformer_tokens, emotional_intensity, emotional_polarity, empathy)
        emotion_examples.append(emotion_item)

    # return all the emotion exmaples
    return emotion_examples

if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.model == "BERT":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = spacy.load("en_core_web_sm")

    # Load train, dev, and test exs and index the words.
    train_exs = parse_dataset(args.train_path, args.model, tokenizer)
    dev_exs = parse_dataset(args.dev_path, args.model, tokenizer)
    test_exs = parse_dataset(args.blind_test_path, args.model, tokenizer)
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples")