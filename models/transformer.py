from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

class BERT(nn.Module):

    def __init__(self, dropout: float, hidden_dim: int, model_name='bert-base-uncased'):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pass
