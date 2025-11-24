from typing import List

from torch import nn
from argparse import Namespace

import evaluation
from emotion_classifier import EmotionExample
from utils import WordEmbeddings
import numpy as np


class Baseline(nn.Module):
    def __init__(self, avg: float):
        super(Baseline, self).__init__()
        self.avg = avg

    def forward(self, x):
        return self.avg

    def predict_all(self, tokens):
        return [self.avg for _ in range(len(tokens))]


def train_baseline(train_exs: List[EmotionExample], dev_exs: List[EmotionExample]):
    intensity = [ex.emotional_intensity for ex in train_exs]
    polarity = [ex.emotional_polarity for ex in train_exs]
    empathy = [ex.empathy for ex in train_exs]

    avg_intensity = np.mean(intensity).item()
    avg_polarity = np.mean(polarity).item()
    avg_empathy = np.mean(empathy).item()

    intensity_model = Baseline(avg_intensity)
    polarity_model = Baseline(avg_polarity)
    empathy_model = Baseline(avg_empathy)

    intensity_mse = evaluation.evaluate_mse(intensity_model, dev_exs, "INTENSITY")
    polarity_mse = evaluation.evaluate_mse(polarity_model, dev_exs, "POLARITY")
    empathy_mse = evaluation.evaluate_mse(empathy_model, dev_exs, "EMPATHY")

train_ex = evaluation.parse_dataset("../data/trac2_CONVT_train.csv", "BASE")
dev_ex = evaluation.parse_dataset("../data/trac2_CONVT_dev.csv", "BASE")
train_baseline(train_ex, dev_ex)