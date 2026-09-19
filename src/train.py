import copy
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    MODEL_PATH,
    HISTORY_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    NUM_WORKERS,
    PIN_MEMORY
)

from src.dataset import (
    create_image_dataframe,
    perform_basic_checks,
    create_data_splits,
    get_train_transform,
    get_evaluation_transform,
    BrainTumorDataset
)

from src.model import BrainTumorCNN

from src.engine import (
    train_one_epoch,
    evaluate_model
)

from src.utils import (
    set_seed,
    get_device,
    validate_project_paths,
    clear_device_cache
)


def calculate_class_weights(
    train_dataframe,
    class_names,
    device
):
    class_counts = (
        train_dataframe[
            "class_name"
        ]
        .value_counts()
        .reindex(class_names)
    )

    class_weights = (
        len(train_dataframe)
        / (
            len(class_names)
            * class_counts.values
        )
    )

    return torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )


def train_model():
    validate_project_paths()
    set_seed(RANDOM_SEED)

    device = get_device()

    print("Device:", device)

    train_full_dataframe = (
        create_image_dataframe(
            TRAIN_DIR
        )
    )

    test_dataframe = (
        create_image_dataframe(
            TEST_DIR
        )
    )

    class_names = perform_basic_checks(
        train_full_dataframe,
        test_dataframe
    )

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(class_names)
    }

    (
        train_dataframe,
        validation_dataframe,
        _
    ) = create_data_splits(
        train_full_dataframe,
        test_dataframe
    )

    train_dataset = BrainTumorDataset(
        dataframe=train_dataframe,
        class_to_index=class_to_index,
        transform=get_train_transform()
    )

    validation_dataset = BrainTumorDataset(
        dataframe=validation_dataframe,
        class_to_index=class_to_index,
        transform=get_evaluation_transform()
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = BrainTumorCNN(
        num_classes=len(class_names),
        dropout=DROPOUT
    ).to(device)

    class_weights = calculate_class_weights(
        train_dataframe,
        class_names,
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )

    best_validation_f1 = -1.0
    best_validation_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    epochs_without_improvement = 0
    history = []

    training_start_time = time.time()

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):
        train_results = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        validation_results = evaluate_model(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device
        )

        validation_macro_f1 = (
            validation_results[
                "macro_f1"
            ]
        )

        scheduler.step(
            validation_macro_f1
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_results["loss"]
                ),
                "train_accuracy": (
                    train_results["accuracy"]
                ),
                "validation_loss": (
                    validation_results[
                        "loss"
                    ]
                ),
                "validation_accuracy": (
                    validation_results[
                        "accuracy"
                    ]
                ),
                "validation_macro_f1": (
                    validation_macro_f1
                ),
                "learning_rate": (
                    current_learning_rate
                )
            }
        )

        

        if (
            validation_macro_f1
            > best_validation_f1
        ):
            best_validation_f1 = (
                validation_macro_f1
            )

            best_validation_loss = (
                validation_results[
                    "loss"
                ]
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        best_model_state
                    ),
                    "class_to_index": {
                        str(class_name): int(
                            class_index
                        )
                        for class_name, class_index
                        in class_to_index.items()
                    },
                    "image_size": int(
                        IMAGE_SIZE
                    ),
                    "dropout": float(
                        DROPOUT
                    ),
                    "learning_rate": float(
                        LEARNING_RATE
                    ),
                    "weight_decay": float(
                        WEIGHT_DECAY
                    ),
                    "best_epoch": int(
                        best_epoch
                    ),
                    "best_validation_loss": float(
                        best_validation_loss
                    ),
                    "best_validation_macro_f1": float(
                        best_validation_f1
                    )
                },
                MODEL_PATH
            )

            print("Best model saved.")

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print("Early stopping activated.")
            break

    training_minutes = (
        time.time()
        - training_start_time
    ) / 60

    history_dataframe = pd.DataFrame(
        history
    )

    history_dataframe.to_csv(
        HISTORY_PATH,
        index=False
    )

    clear_device_cache(device)

    print(
        f"Training completed in "
        f"{training_minutes:.2f} minutes."
    )

    print("Best epoch:", best_epoch)

    print(
        "Best validation macro F1:",
        round(
            best_validation_f1,
            4
        )
    )

    print(
        "Saved model:",
        MODEL_PATH.resolve()
    )


if __name__ == "__main__":
    train_model()