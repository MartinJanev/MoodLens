import torch

@torch.no_grad()
def accuracy(logits, targets):
    """
    Computes the accuracy of predictions against targets.
    :param logits: Tensor of shape (batch_size, num_classes) containing model predictions.
    :param targets: Tensor of shape (batch_size,) containing true labels.
    :return: Accuracy as a float value.
    """
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()
