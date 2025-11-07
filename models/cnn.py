from argparse import Namespace
from typing import List
import torch
import torch.nn as nn
from torch import optim
from emotion_classifier import EmotionExample
from utils import *
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class CNN(nn.Module):
    """
    Subclass nn.Module for CNN initialization
    """

    def __init__(self, inp: int, num_filters: int, dropout: float):
        super(CNN, self).__init__()
        self.filter_sizes = [3, 4, 5]

        self.conv_list = nn.ModuleList()

        for filter_size in self.filter_sizes:
            self.conv_list.append(nn.Conv1d(in_channels=inp, out_channels=num_filters, kernel_size=filter_size))

        self.dropout = nn.Dropout(dropout)
        self.fully_connected_layer = nn.Linear(num_filters * len(self.filter_sizes), 1)

    def forward(self, x):
        conv_outputs = []
        for conv in self.conv_list:
            conv_outputs.append(torch.max(torch.relu(conv(x)), dim=2)[0])

        output = torch.cat(conv_outputs, dim=1)

        output = self.dropout(output)
        output = self.fully_connected_layer(output)

        return output
    
class EmotionClassifierCNN(object):
    """
    EmotionClassifier for the CNN model
    """

    def __init__(self, word_embeddings: WordEmbeddings, num_filters: int, dropout: float, batch_size: int):
        """
        Initialize the embeddings, embedding layer, CNN model, and batch size for emotion classification
        """

        self.word_embeddings = word_embeddings
        self.embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)

        self.model = CNN(word_embeddings.get_embedding_length(), num_filters, dropout)
        self.batch_size = batch_size

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        Return a list of model predictions using the trained CNN
        """

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(all_ex_words), self.batch_size):
                batch = all_ex_words[i:i+self.batch_size]
                sentences = []

                for example in batch:
                    ex_indices = []
                    for ex_word in example:
                        ex_word_idx = self.word_embeddings.word_indexer.index_of(ex_word)
                        if ex_word_idx == -1:
                            ex_word_idx = self.word_embeddings.word_indexer.index_of("UNK")
                        ex_indices.append(ex_word_idx)
                    sentences.append(ex_indices)

                max_sentence_length = max(len(sentence) for sentence in sentences)
                padded_sentences = torch.LongTensor([sentence + [0] * (max_sentence_length - len(sentence)) for sentence in sentences])
                embeddings = self.embedding_layer(padded_sentences)

                outputs = self.model.forward(embeddings.permute(0, 2, 1)).squeeze(1)

                for output in outputs:
                    predictions.append(output.item())
        
        return predictions
    
def train_CNN(args: Namespace, train_exs: List[EmotionExample], dev_exs: List[EmotionExample],
              word_embeddings: WordEmbeddings, target: str) -> EmotionClassifierCNN:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained EmotionClassifierCNN model
    """

    cnn = EmotionClassifierCNN(word_embeddings, args.hidden_dim, args.dropout, args.batch_size)

    cnn.model.train()

    optimizer = optim.AdamW(cnn.model.parameters(), args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    
    train_losses = []
    dev_losses = []

    for epoch in range(args.num_epochs):
        random.shuffle(train_exs)
        total_loss = 0.0
        for i in range(0, len(train_exs), args.batch_size):
            batch = train_exs[i:i+args.batch_size]
            sentences = []
            labels = []

            for example in batch:
                ex_indices = []
                for ex_word in example.tokens:
                    ex_word_idx = word_embeddings.word_indexer.index_of(ex_word)
                    if ex_word_idx == -1:
                        ex_word_idx = word_embeddings.word_indexer.index_of("UNK")
                    ex_indices.append(ex_word_idx)
                sentences.append(ex_indices)
                
                if target == "EMPATHY":
                    label = example.empathy
                elif target == "POLARITY":
                    label = example.emotional_polarity
                else:
                    label = example.emotional_intensity

                labels.append(label)

            max_sentence_length = max(len(sentence) for sentence in sentences)
            padded_sentences = torch.LongTensor([sentence + [0] * (max_sentence_length - len(sentence)) for sentence in sentences])
            labels = torch.FloatTensor(labels)
            embeddings = cnn.embedding_layer(padded_sentences)

            cnn.model.zero_grad()
            outputs = cnn.model.forward(embeddings.permute(0, 2, 1)).squeeze(1)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_losses.append(total_loss)

        cnn.model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(dev_exs), args.batch_size):
                batch = dev_exs[i:i+args.batch_size]
                sentences = []
                labels = []

                for example in batch:
                    ex_indices = []
                    for ex_word in example.tokens:
                        ex_word_idx = word_embeddings.word_indexer.index_of(ex_word)
                        if ex_word_idx == -1:
                            ex_word_idx = word_embeddings.word_indexer.index_of("UNK")
                        ex_indices.append(ex_word_idx)
                    sentences.append(ex_indices)

                    if target == "EMPATHY":
                        label = example.empathy
                    elif target == "POLARITY":
                        label = example.emotional_polarity
                    else:
                        label = example.emotional_intensity

                    labels.append(label)

                    

                max_sentence_length = max(len(sentence) for sentence in sentences)
                padded_sentences = torch.LongTensor([sentence + [0] * (max_sentence_length - len(sentence)) for sentence in sentences])
                labels = torch.FloatTensor(labels)
                embeddings = cnn.embedding_layer(padded_sentences)

                outputs = cnn.model.forward(embeddings.permute(0, 2, 1)).squeeze(1)

                loss = loss_fn(outputs, labels)
                dev_loss += loss.item()

        dev_losses.append(dev_loss)
        cnn.model.train()

        print("Total loss on epoch %i: %f" % (epoch + 1, total_loss))
    
    plt.plot(range(1, args.num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, args.num_epochs+1), dev_losses, label="Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(target + " CNN Training vs Dev Loss")
    plt.legend()
    plt.show()

    return cnn