from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch import optim
from typing import List
from argparse import Namespace
from emotion_classifier import EmotionExample
import random

random.seed(42)
torch.manual_seed(42)

class BERT(nn.Module):
    """
    BERT-based regression model for emotion prediction
    """
    def __init__(self, dropout: float, hidden_dim: int, model_name='bert-base-uncased'):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Regression head
        # Architecture: BERT output (768) -> Hidden layer (100) -> Output (1)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass through BERT model

        :param input_ids: Token IDs from BERT tokenizer [batch_size, seq_len]
        :param attention_mask: Attention mask for padding [batch_size, seq_len]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # using [CLS] token for sentence-level tasks during BERT pre-training
        cls_representation = outputs.last_hidden_state[:,0,:] # [batch_size, hidden_size]
        predictions = self.regression_head(cls_representation) # [batch_size, 1]
        return predictions.squeeze(1) # [batch_size]
    
class EmotionClassifierBert(object):
    """
    Emotion classifier using BERT
    """
    def __init__(self, hidden_dim: int, dropout: float, batch_size: int):
        """
        Initialize BERT-based emotion classifier

        :param hidden_dim: Size of hidden layer in regression head
        :param dropout: Dropout probability for regularization
        :param batch_size: Batch size for training and inference
        """
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BERT(dropout, hidden_dim, 'bert-base-uncased')

    def predict_all(self, all_ex_words: List[List[str]]) -> List[float]:
        """
        Make predictions for all examples

        :param all_ex_words: List of tokenized sentences (list of word lists)
        :return: List of predicted values
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(all_ex_words), self.batch_size):
                batch = all_ex_words[i:i+self.batch_size]

                # joining tokens back into sentences for BERT tokenization
                sentences = [' '.join(tokens) for tokens in batch]

                # tokenize with BERT tokenizer
                encoded = self.tokenizer(
                    sentences, 
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt' # to retun PyTorch tensors
                )

                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']

                outputs = self.model(input_ids, attention_mask)
                predictions.extend(outputs.tolist())
        
        return predictions
    
def train_BERT(args: Namespace, train_exs: List[EmotionExample], dev_exs: List[EmotionExample], target: str) -> EmotionClassifierBert:
    """
    Train a BERT-based emotion classifier

    :param args: Command-line arguments containing hyperparameters
    :param train_exs: Training examples
    :param dev_exs: Development examples for validation
    :param target: Target variables to predict (EMPATHY, POLARITY, or INTENSITY)
    :return: Trained EmotionClassifierBERT
    """
    print(f"Training BERT for {target} prediction")

    # initialize classifier
    classifier = EmotionClassifierBert(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size
    )

    # set model to training mode
    classifier.model.train()

    # hyperparameters:
    #   lr = 2e-5 (standard value for BERT fine-tuning, higher values can cause catastrophic forgetting)
    #   weight decay = 0.01
    #   L2 regularization to prevent overfitting
    #   AdamW optimizer decouples weight decay from gradients (better than Adam)
    optimizer = optim.AdamW(
        classifier.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Loss Function: using SmoothL1Loss (Huber loss) which is robust to outliers
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(args.num_epochs):
        random.shuffle(train_exs)
        total_loss = 0.0

        for i in range(0, len(train_exs), args.batch_size):
            batch = train_exs[i:i+args.batch_size]
            sentences = [' '.join(ex.tokens) for ex in batch]

            if target == "EMPATHY":
                labels = torch.FloatTensor([ex.empathy for ex in batch])
            elif target == "POLARITY":
                labels = torch.FloatTensor([ex.emotional_polarity for ex in batch])
            else:
                labels = torch.FloatTensor([ex.emotional_intensity for ex in batch])

            encoded = classifier.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

            classifier.model.zero_grad()
            predictions = classifier.model(input_ids, attention_mask)

            # loss calculation
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()

            # backward pass and optimization
            loss.backward()

            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), max_norm=1.0)

            optimizer.step()
        print(f"Epoch {epoch + 1}/{args.num_epochs} - Train Loss: {total_loss:.4f}")

    return classifier