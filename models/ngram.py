import torch
import torch.nn as nn 
from torch import optim
from typing import List, Tuple
import random
from utils import Indexer
from emotion_classifier import EmotionExample

random.seed(42)
torch.manual_seed(42)

def create_ngrams(tokens: List[str], n: int) -> List[Tuple[str]]:
  """
    Take a sequence of tokens and return a list of n-grams
  """
  n_grams = []

  for i in range(0, len(tokens)):
      if i + n > len(tokens) - 1: break
      upper_bound = min(len(tokens)  - 1, i + n)

      n_grams.append(tuple(tokens[i:upper_bound]))
  return n_grams

def ngram_to_index_tensors(n_grams: List[Tuple[str]], indexer: Indexer, add_words: bool):
    """
    Converts a set of n-grams to a single list of the embedding indices of each token concatenated together in order
    """
    indicies = []
    # for each n gram
    for n_gram in n_grams:
        # iterate through all the tokens
        for token in n_gram:
            # and generate a tuple of the embedding indicies
            #gram_index = []
            if not add_words and not indexer.contains(token):
                # if the word is not in the dictionary, return index of <UNK> (1)
                indicies.append(1)
            else:
                indicies.append(indexer.add_and_get_index(token, add_words))
        #indicies.append(torch.tensor(indicies))
    return indicies


def create_batches(train_exs: List[EmotionExample], indexer: Indexer, batch_size: int, max_embedding_length: int):
    """
    Converts the given list of training examples to a set of batches mapping token indexes to class gold values
    """
    #initialize the batches
    batches = [([], [])]
    current_batch = 0

    for ex in train_exs:
        tokens = ex.tokens
        indices = ngram_to_index_tensors(tokens, indexer, False)
        if len(indices) > max_embedding_length:
            indices = indices[0 : max_embedding_length]
        else:
            padding = [0] * (max_embedding_length - len(indices))
            indices.extend(padding)
        # populate the batch tiple with the sentence indices and label
        #TODO: paramaterize which label we want to use here
        batches[current_batch][0].append(indices)
        batches[current_batch][1].append(ex.emotionoal_intensity) # for example

        #If this batch is full, ad a new empty batch and start populating that one
        if len(batches[current_batch][1]) >= batch_size:
            current_batch += 1
            batches.append(([], []))
    
    # remove the last batch if it is empty
    if len(batches[current_batch][1]) == 0:
        batches.pop()
    
    return batches

class Ngram(nn.Model):
    def __init__(self, n: int):
        self.n = n

        self.embedding_dim = ...
        self.num_embeddings = ...#|V|

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.model = nn.Sequential(
            #other layers
            nn.Linear(self.embedding_dim, 1)
        )

    def forward(self, x):
        # convert x to a tensor if it isn't already
        x = torch.tensor(x)

        embeddings = self.embedding(x)
        # taking the average to use in DAN...could instead classify each n-gram seperately and average the result. Thinking on it...
        average_embedding = torch.mean(embeddings, dim = 1)

        return self.model(average_embedding)
    



def train_ngram_network(train_exs: List[EmotionExample], dev_exs: List[EmotionExample], num_epochs: int, n: int, batch_size: int):
    ffnn = Ngram(n)
    indexer = Indexer()

    # build vocabulary
    for ex in train_exs:
        for token in ex.tokens:
            indexer.add_and_get_index(token, add=True)
    
    # define our loss function and separate data into batches
    loss_fn = nn.MSELoss()
    batches = create_batches(train_exs, batch_size, indexer)

    ffnn.train()
    lr = .001
    optimizer = optim.Adam(ffnn.parameters(), lr)

    loss_per_epoch = []

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(batches))]
        random.shuffle(ex_indices)

        total_loss - 0.0
        for idx in ex_indices:
            # get this input and its corresponding label
            x = batches[idx][0]
            y = batches[idx][1]

            # Zero out the gradients from the FFNN object
            ffnn.zero_grad()
            prediction = ffnn.forward(x)

            # Calculate the loss
            y_tensor = torch.tensor(y, dtype=torch.long) # gold
            loss = loss_fn(prediction, y_tensor)

            # run backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # evaluate model on dev set here if we want to plot loss over dev set

        loss_per_epoch.append(total_loss)
        print(f"Total loss on epoch {epoch}: {total_loss}")
    
    ffnn.eval()
    #TODO: change this to a wrapper class. Should probaby either make an interface for all thre models to implement
    # or return the same wrapper for all three
    return ffnn

