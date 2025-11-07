from argparse import Namespace
import torch
import torch.nn as nn 
from torch import optim
from typing import List, Tuple
import random
from utils import Indexer, WordEmbeddings
from emotion_classifier import EmotionExample

random.seed(42)
torch.manual_seed(42)

def create_ngrams(tokens: List[str], n: int) -> List[Tuple[str]]:
  """
    Take a sequence of tokens and return a list of n-grams
  """
  n_grams = []

  for i in range(0, len(tokens)):
      if i + n > len(tokens): break
      upper_bound = min(len(tokens), i + n)

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


def create_batches(train_exs: List[EmotionExample], indexer: Indexer, batch_size: int, max_embedding_length: int, n: int, target: str):
    """
    Converts the given list of training examples to a set of batches mapping token indexes to class gold values
    """
    #initialize the batches
    batches = [([], [])]
    current_batch = 0

    for ex in train_exs:
        n_grams = create_ngrams(ex.tokens, n)
        indices = ngram_to_index_tensors(n_grams, indexer, False)
        if len(indices) > max_embedding_length:
            indices = indices[0 : max_embedding_length]
        else:
            padding = [0] * (max_embedding_length - len(indices))
            indices.extend(padding)
        # populate the batch tiple with the sentence indices and label
        batches[current_batch][0].append(indices)
        if target == "EMPATHY":
            batches[current_batch][1].append(ex.empathy)
        elif target == "POLARITY":
            batches[current_batch][1].append(ex.emotional_polarity)
        else:
            batches[current_batch][1].append(ex.emotional_intensity)

        #If this batch is full, ad a new empty batch and start populating that one
        if len(batches[current_batch][1]) >= batch_size:
            current_batch += 1
            batches.append(([], []))
    
    # remove the last batch if it is empty
    if len(batches[current_batch][1]) == 0:
        batches.pop()
    
    return batches

class Ngram(nn.Module):
    def __init__(self, embedding_dim: int, embedding_layer: nn.Embedding, num_embeddings: int, hidden_dim: int, n: int, dropout: float):
        super(Ngram, self).__init__()
        self.n = n

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding_layer = embedding_layer
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        # taking the average to use in DAN...could instead classify each n-gram seperately and average the result. Thinking on it...
        average_embedding = torch.mean(embeddings, dim = 1)

        return self.model(average_embedding)
    

class EmotionClassifierNgram(object):
    def __init__(self, word_embeddings: WordEmbeddings,  indexer: Indexer, hidden_dim: int, n: int, dropout: float) -> None:
        self.word_embeddings = word_embeddings
        self.embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)

        self.model = Ngram(word_embeddings.get_embedding_length(), self.embedding_layer, len(indexer), hidden_dim, n, dropout)

        self.indexer = indexer
        self.n = n

    def predict(self, ex_words):
        with torch.no_grad():
            n_grams = create_ngrams(ex_words, self.n)
            index_tensor = ngram_to_index_tensors(n_grams, self.indexer, False)
            x = torch.tensor(index_tensor, dtype=torch.long).unsqueeze(0)
            return self.model.forward(x).item()
        
    def predict_all(self, all_ex_words: List[List[str]]):
        with torch.no_grad():
            all_n_grams = list(map(lambda sentence: create_ngrams(sentence, self.n), all_ex_words))
            index_batch = torch.tensor(map(lambda n_grams: ngram_to_index_tensors(n_grams, False), all_n_grams))

            return self.model.forward(index_batch).tolist()


def train_ngram_network(args: Namespace, train_exs: List[EmotionExample], dev_exs: List[EmotionExample],
                        word_embeddings: WordEmbeddings, target: str):
    #ffnn = Ngram(args.n_grams)
    indexer = Indexer()

    # build vocabulary
    for ex in train_exs:
        for token in ex.tokens:
            indexer.add_and_get_index(token, add=True)

    ffnn = EmotionClassifierNgram(word_embeddings, indexer, args.hidden_dim, args.n_grams, args.dropout)
    
    # define our loss function and separate data into batches
    loss_fn = nn.SmoothL1Loss()
    batches = create_batches(train_exs, indexer, args.batch_size, args.max_sequence_len, args.n_grams, target)

    ffnn.model.train()
    optimizer = optim.AdamW(ffnn.model.parameters(), args.lr, weight_decay=args.weight_decay)

    loss_per_epoch = []

    for epoch in range(0, args.num_epochs):
        ex_indices = [i for i in range(0, len(batches))]
        random.shuffle(ex_indices)

        total_loss = 0.0
        for idx in ex_indices:
            # get this input and its corresponding label
            x = batches[idx][0]
            y = batches[idx][1]

            # Zero out the gradients from the FFNN object
            ffnn.model.zero_grad()
            prediction = ffnn.model.forward(torch.tensor(x, dtype=torch.long))

            # Calculate the loss
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # gold
            loss = loss_fn(prediction, y_tensor)

            # run backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # evaluate model on dev set here if we want to plot loss over dev set

        loss_per_epoch.append(total_loss)
        print(f"Total loss on epoch {epoch}: {total_loss}")
    
    ffnn.model.eval()
    return ffnn

