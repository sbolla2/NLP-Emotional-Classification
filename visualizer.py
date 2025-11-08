import math

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Optional


def scatter(predictions, true_labels, best_fit = True, model_name: str = ""):
    """
    Generates a scatterplot of predicted values vs true values
    :param model_name:
    :param predictions: Values predicted by the model
    :param true_labels: True labels for the predicted data
    :param best_fit: Should the plot include a line of best fit
    """
    plt.scatter(predictions, true_labels)
    plt.xlabel('Prediction')
    plt.ylabel('True label')
    plt.title(f"{model_name} Prediction vs. True Label")
    if best_fit:
        # perform a linear regression and get a line p(x)
        coef = np.polyfit(predictions, true_labels, 1)
        p = np.poly1d(coef)

        plt.plot(predictions, p(predictions), 'b')
    plt.show()

def loss_over_epochs(loss: List[float], dev_loss: bool = False, model_name: str = ""):
    """
    Simply graphs the given loss function as a line with appropriate labels
    :param model_name: Name of the model to add to title
    :param dev_loss: Set to true if this is dev loss to change title
    :param loss: Loss values from training
    """
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if dev_loss:
        plt.title('Dev Loss Over Epochs')
    else:
        plt.title(f"{model_name} Training Loss Over Epochs")

    plt.show()

def bar_graphs(pearsons: List[float], model_names: Optional[List[str]] = None):
    """
    Generates a bar graph with a bar for each of the given pearson correlation values
    :param pearsons: Pearson correlations values for various models
    :param model_names: Names of the models corresponding to the given pearson values
    """
    fix, ax = plt.subplots()

    x_pos = np.arange(len(pearsons))

    # set color depending on correlation
    colors = ['green' if c >=0 else 'red' for c in pearsons]
    bars = ax.bar(x_pos, pearsons, color=colors, alpha=0.7, ecolor='black', linewidth=1.5)

    ax.set_xlabel('Models', weight='bold')
    ax.set_ylabel('Pearson Correlation Coefficient', weight='bold')
    ax.set_title('Comparison of Pearson Correlation Coefficients Across Models', weight='bold')

    # set the range of the bars to the range of pearson coefficient
    ax.set_ylim([-1, 1])
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

    # put model names below bars if available
    ax.set_xticks(x_pos)
    if model_names:
        ax.set_xticklabels(model_names)

    # Add value labels on top of each bar
    for bar, value in zip(bars, pearsons):
        height = bar.get_height()
        # Position label above bar if positive, below if negative
        y_pos = height + 0.02 if height >= 0 else height - 0.05
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos, f'{value:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

    # Add background colors for flair
    ax.axhspan(-1, -0.5, alpha=0.1, color='red', label='Strong Negative')
    ax.axhspan(-0.5, 0, alpha=0.05, color='red', label='Weak Negative')
    ax.axhspan(0, 0.5, alpha=0.05, color='green', label='Weak Positive')
    ax.axhspan(0.5, 1, alpha=0.1, color='green', label='Strong Positive')

    plt.tight_layout()
    plt.show()

# run to see what the models look like
def test_visualizations():
    random_predictions = np.random.random(100)
    random_gold_labels = np.random.random(100)
    scatter(random_predictions, random_gold_labels,True, "Test Model")

    nums = np.arange(20)
    loss_over_epochs(list(map(lambda x : 1/(x + 1), nums)), False, "Test Model")

    bar_graphs([0.75, -0.45, 0.92, -0.3], ['Model A', 'Model B', 'Model C', 'Model D'])

test_visualizations()