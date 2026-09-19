import numpy as np
import torch

from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    progress_bar = tqdm(
        data_loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        total_loss
        / len(data_loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return {
        "loss": float(epoch_loss),
        "accuracy": float(epoch_accuracy)
    }


def evaluate_model(
    model,
    data_loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc="Evaluating",
            leave=False
        ):
            if len(batch) == 3:
                images, labels, paths = batch
                all_paths.extend(paths)
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    labels_array = np.array(
        all_labels
    )

    predictions_array = np.array(
        all_predictions
    )

    probabilities_array = np.array(
        all_probabilities
    )

    accuracy = accuracy_score(
        labels_array,
        predictions_array
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        average="macro",
        zero_division=0
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _
    ) = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        average="weighted",
        zero_division=0
    )

    return {
        "loss": float(
            total_loss
            / len(data_loader.dataset)
        ),
        "accuracy": float(accuracy),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "labels": labels_array,
        "predictions": predictions_array,
        "probabilities": probabilities_array,
        "paths": all_paths
    }