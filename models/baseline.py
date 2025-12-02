from typing import List
from torch import nn
import evaluation
import utils
from emotion_classifier import EmotionExample
import numpy as np

class Baseline(nn.Module):
    """
    A simple baseline model that always predicts the average value from the training set for any input
    """
    def __init__(self, avg: float):
        super(Baseline, self).__init__()
        self.avg = avg

    def forward(self, x):
        """
        Args:
            x: The EmotionExample input

        Returns: The average value of the target in the training set
        """
        return self.avg

    def predict_all(self, tokens):
        """
        Args:
            tokens: A list of EmotionExample inputs

        Returns: A list of floats the same length as the inputted list where each value is the
        average of the target in the training set
        """
        return [self.avg for _ in range(len(tokens))]


def train_baseline(train_exs: List[EmotionExample], dev_exs: List[EmotionExample]):
    """
    Trains a baseline model for each target by calculating the average of each
    of the targets in the training set

    Args:
        train_exs: A list of EmotionExamples to calculate the average values from
        dev_exs: A list of EmotionExamples to use to evaluate the created baseline models
    """

    # extract the targets from the training examples
    intensity = [ex.emotional_intensity for ex in train_exs]
    polarity = [ex.emotional_polarity for ex in train_exs]
    empathy = [ex.empathy for ex in train_exs]

    # calculate the averages of each of the targets
    avg_intensity = np.mean(intensity).item()
    avg_polarity = np.mean(polarity).item()
    avg_empathy = np.mean(empathy).item()

    # initialize each of the models
    intensity_model = Baseline(avg_intensity)
    polarity_model = Baseline(avg_polarity)
    empathy_model = Baseline(avg_empathy)

    # evaluate each of the models
    intensity_mse = evaluation.evaluate_mse(intensity_model, dev_exs, "INTENSITY")
    polarity_mse = evaluation.evaluate_mse(polarity_model, dev_exs, "POLARITY")
    empathy_mse = evaluation.evaluate_mse(empathy_model, dev_exs, "EMPATHY")

    # save the models to re-evaluate later
    utils.save_model(intensity_model, "baseline/intensity")
    utils.save_model(polarity_model, "baseline/polarity")
    utils.save_model(empathy_model, "baseline/empathy")

train_ex = evaluation.parse_dataset("../data/trac2_CONVT_train.csv", "BASE")
dev_ex = evaluation.parse_dataset("../data/trac2_CONVT_dev.csv", "BASE")
train_baseline(train_ex, dev_ex)