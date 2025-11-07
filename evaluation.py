import argparse
from models.cnn import train_CNN
from typing import List
from scipy.stats import pearsonr
from emotion_classifier import EmotionExample
import pandas as pd
import spacy
from transformers import AutoTokenizer
from collections import Counter
from utils import *
import math

tokenizer = spacy.load("en_core_web_sm")
transformer_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='CNN', help='model to run (NGRAM, CNN, or BERT)')
    parser.add_argument('--target', type=str, default='EMPATHY', help='target to predict (EMPATHY, POLARITY, or INTENSITY)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden dimensions for N-Grams or filters for CNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for CNN')
    parser.add_argument('--batch_size', type=int, default=50, help='training batch size; 50 by default')
    parser.add_argument('--n_grams', type=int, default=3, help='no. of n-grams')
    parser.add_argument('--max_sequence_len', type=int, default=50, help='max sequence length for n-grams')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')

    # paths to data
    parser.add_argument('--train_path', type=str, default="data/trac2_CONVT_train.csv", help='model training data')
    parser.add_argument('--dev_path', type=str, default="data/trac2_CONVT_dev.csv", help='model development data')
    parser.add_argument('--test_path', type=str, default="data/goldstandard_CONVT.csv", help='model validation data')
    parser.add_argument('--embeddings_path', type=str, default="data/glove.6B.300d-relativized.txt", help='relativized embeddings')

    args = parser.parse_args()
    return args

def parse_dataset(data_path: str, model: str):
    """
    Given a path to a csv file containing training or test data, converts that data to a list of 
    EmotionExmples and returns the resulting list 
    """

    # read data as pandas dataframe
    df = pd.read_csv(data_path, engine='python', escapechar='\\')
    emotion_examples = []

    # iterate through dataframe and create EmotionExample object for each
    for _, row in df.iterrows(): 
        sentence = row["text"]

        doc = tokenizer(sentence)              # run spaCy on the sentence
        tokens = [token.text for token in doc]   # extract token strings

        transformer_tokens = None
        if model == "BERT":
            transformer_tokens = transformer_tokenizer(sentence, return_tensors=None)
        
        emotional_intensity = float(row["Emotion"])
        emotional_polarity = float(row["EmotionalPolarity"])
        empathy = float(row["Empathy"])
        # other row are just ids, probably won't need them

        if (math.isnan(emotional_intensity) or 
            math.isnan(emotional_polarity) or 
            math.isnan(empathy)):
            continue

        emotion_item = EmotionExample(tokens, transformer_tokens, emotional_intensity, emotional_polarity, empathy)
        emotion_examples.append(emotion_item)

    # return all the emotion exmaples
    return emotion_examples

def evaluate(model, exs: List[EmotionExample], target: str):
    # Extract the list of token lists from the examples
    all_tokens = [ex.tokens for ex in exs]

    # Get predictions (a list of floats)
    preds = model.predict_all(all_tokens)

    # Extract true labels based on target
    if target == "EMPATHY":
        true_labels = [ex.empathy for ex in exs]
    elif target == "POLARITY":
        true_labels = [ex.emotional_polarity for ex in exs]
    elif target == "INTENSITY":
        true_labels = [ex.emotional_intensity for ex in exs]
    else:
        raise ValueError(f"Unknown target {target}")

    pearson_corr = pearsonr(true_labels, preds)[0]
    print(f"Pearson correlation on {target}: {pearson_corr:.4f}")
    return pearson_corr

def relativize(file, outfile, word_counter):
    """
    Relativize the word vectors to the given dataset represented by word counts
    :param file: word vectors file
    :param outfile: output file
    :param word_counter: Counter of words occurring in train/dev/test data
    :return:
    """
    f = open(file)
    o = open(outfile, 'w')
    voc = []
    for line in f:
        word = line[:line.find(' ')]
        if word_counter.get(word, 0) > 0:
            # print("Keeping word vector for " + word)
            voc.append(word)
            o.write(line)
    for word in word_counter:
        if word not in voc:
            count = word_counter[word]
            if count > 1:
                print("Missing " + word + " with count " + repr(count))
    f.close()
    o.close()


def relativize_emotion_data(args: argparse.Namespace):
    # Count all words in the train, dev, and *test* sets. Note that this use of looking at the test set is legitimate
    # because we're not looking at the labels, just the words, and it's only used to cache computation that we
    # otherwise would have to do later anyway.
    word_counter = Counter()
    for ex in parse_dataset(args.train_path, args.model):
        for word in ex.tokens:
            word_counter[word] += 1
    for ex in parse_dataset(args.dev_path, args.model):
        for word in ex.tokens:
            word_counter[word] += 1
    for ex in parse_dataset(args.test_path, args.model):
        for word in ex.tokens:
            word_counter[word] += 1

    relativize("data/glove.6B.300d.txt", "data/glove.6B.300d-relativized.txt", word_counter)

if __name__ == '__main__':
    args = parse_args()
    print(args)

    # Already relativized the GloVe embeddings, so need
    #relativize_emotion_data(args)

    # Load train, dev, and test exs and index the words.

    train_exs = parse_dataset(args.train_path, args.model)
    dev_exs = parse_dataset(args.dev_path, args.model)
    test_exs = parse_dataset(args.test_path, args.model)
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples")

    word_embeddings = read_word_embeddings(args.embeddings_path)

    cnn_model = train_CNN(args, train_exs, dev_exs, word_embeddings, args.target)
    
    print("Pearson Correlation for Train Set:")
    train_corr = evaluate(cnn_model, train_exs, args.target)
    print("\nPearson Correlation for Dev Set:")
    dev_corr = evaluate(cnn_model, dev_exs, args.target)
    print("\nPearson Correlation for Test Set:")
    test_corr = evaluate(cnn_model, test_exs, args.target)

    average_corr = (train_corr + dev_corr + test_corr) / 3
    print(f"\nAverage Pearson Correlation across Train, Dev, and Test: {average_corr:.4f}")