import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from src.config import (
    TEST_DIR,
    FINAL_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    TEST_PREDICTIONS_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    CONFIDENCE_THRESHOLD
)

from src.dataset import (
    create_image_dataframe,
    get_evaluation_transform,
    BrainTumorDataset
)

from src.engine import evaluate_model
from src.model import BrainTumorCNN

from src.utils import (
    get_device,
    validate_project_paths
)


def load_checkpoint(
    checkpoint_path,
    device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    class_to_index = checkpoint[
        "class_to_index"
    ]

    class_names = [
        class_name
        for class_name, class_index
        in sorted(
            class_to_index.items(),
            key=lambda item: item[1]
        )
    ]

    dropout = float(
        checkpoint.get(
            "dropout",
            0.40
        )
    )

    model = BrainTumorCNN(
        num_classes=len(class_names),
        dropout=dropout
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        class_to_index,
        class_names,
        checkpoint
    )


def evaluate_saved_model():
    validate_project_paths()

    device = get_device()

    (
        model,
        class_to_index,
        class_names,
        checkpoint
    ) = load_checkpoint(
        FINAL_MODEL_PATH,
        device
    )

    test_dataframe = create_image_dataframe(
        TEST_DIR
    )

    test_dataset = BrainTumorDataset(
        dataframe=test_dataframe,
        class_to_index=class_to_index,
        transform=get_evaluation_transform(),
        return_path=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    criterion = nn.CrossEntropyLoss()

    results = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(
        "Test loss:",
        round(results["loss"], 4)
    )

    print(
        "Test accuracy:",
        round(results["accuracy"], 4)
    )

    print(
        "Macro precision:",
        round(
            results["macro_precision"],
            4
        )
    )

    print(
        "Macro recall:",
        round(
            results["macro_recall"],
            4
        )
    )

    print(
        "Macro F1:",
        round(
            results["macro_f1"],
            4
        )
    )

    print(
        "Weighted F1:",
        round(
            results["weighted_f1"],
            4
        )
    )

    report = classification_report(
        results["labels"],
        results["predictions"],
        labels=list(
            range(len(class_names))
        ),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_dataframe = pd.DataFrame(
        report
    ).transpose()

    report_dataframe.to_csv(
        CLASSIFICATION_REPORT_PATH
    )

    confusion_values = confusion_matrix(
        results["labels"],
        results["predictions"],
        labels=list(
            range(len(class_names))
        )
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        confusion_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    index_to_class = {
        class_index: class_name
        for class_name, class_index
        in class_to_index.items()
    }

    confidences = results[
        "probabilities"
    ].max(axis=1)

    prediction_dataframe = pd.DataFrame(
        {
            "image_path": results["paths"],
            "actual_label": results[
                "labels"
            ],
            "predicted_label": results[
                "predictions"
            ],
            "actual_class": [
                index_to_class[int(label)]
                for label
                in results["labels"]
            ],
            "predicted_class": [
                index_to_class[int(label)]
                for label
                in results["predictions"]
            ],
            "confidence": confidences
        }
    )

    prediction_dataframe[
        "is_correct"
    ] = (
        prediction_dataframe[
            "actual_label"
        ]
        == prediction_dataframe[
            "predicted_label"
        ]
    )

    prediction_dataframe[
        "requires_review"
    ] = (
        prediction_dataframe[
            "confidence"
        ]
        < CONFIDENCE_THRESHOLD
    )

    prediction_dataframe.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False
    )

    print(
        "Classification report saved to:",
        CLASSIFICATION_REPORT_PATH.resolve()
    )

    print(
        "Confusion matrix saved to:",
        CONFUSION_MATRIX_PATH.resolve()
    )

    print(
        "Test predictions saved to:",
        TEST_PREDICTIONS_PATH.resolve()
    )


if __name__ == "__main__":
    evaluate_saved_model()