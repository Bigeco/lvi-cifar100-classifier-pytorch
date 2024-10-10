import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def correct_class_counts(true_labels, preds):
    correct_pred = np.array([0 for _ in range(100)])

    _, top_k_preds = preds.topk(1, dim=1, largest=True, sorted=True)
    correct = top_k_preds.eq(true_labels.view(-1, 1).expand_as(top_k_preds))
    unique_values = true_labels.unique()

    for idx in unique_values:
        indices = (true_labels == idx).nonzero().flatten()
        value = int(correct[indices.tolist()].float().sum())
        correct_pred[idx] += value
    return correct_pred


def plot_correct_class_counts(correct_pred):
    plt.figure(figsize=(20, 10))
    sns.set_palette("husl", n_colors=100)
    plt.title("Number of Correct Predictions per Label")
    sns.barplot(x=list(range(100)), y=correct_pred, palette="Spectral")
    plt.xlabel("Label")
    plt.ylabel("Number of Correct Predictions")
    plt.tight_layout()
    plt.show()


def plot_loss_curves(lst_epoch_train_loss, lst_epoch_valid_loss):
    plt.figure(figsize=(14,6))

    plt.title("Train and Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    loss_df = pd.DataFrame({
        'train loss': lst_epoch_train_loss,
        'valid loss': lst_epoch_valid_loss,
        'index': range(1, len(lst_epoch_valid_loss) + 1)
    })
    loss_df.set_index('index', inplace=True)

    sns.lineplot(data=loss_df)
    plt.show()


# TODO 1: 다음 아래 명령어가 가능하도록 할 것
# python visualizations.py --model resnet --plot-type loss