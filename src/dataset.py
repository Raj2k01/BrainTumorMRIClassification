from pathlib import Path

import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

from src.config import (
    IMAGE_SIZE,
    VALIDATION_SIZE,
    RANDOM_SEED,
    SUPPORTED_EXTENSIONS
)


def create_image_dataframe(directory):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory.resolve()}"
        )

    image_records = []

    class_directories = sorted(
        [
            folder
            for folder in directory.iterdir()
            if folder.is_dir()
            and not folder.name.startswith(".")
        ]
    )

    if not class_directories:
        raise ValueError(
            f"No class folders found in "
            f"{directory.resolve()}"
        )

    for class_directory in class_directories:
        class_name = class_directory.name

        for image_path in class_directory.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                image_records.append(
                    {
                        "image_path": str(
                            image_path.resolve()
                        ),
                        "class_name": class_name
                    }
                )

    dataframe = pd.DataFrame(
        image_records
    )

    if dataframe.empty:
        raise ValueError(
            f"No supported images found in "
            f"{directory.resolve()}"
        )

    return dataframe


def perform_basic_checks(
    train_dataframe,
    test_dataframe
):
    train_missing = int(
        train_dataframe
        .isnull()
        .sum()
        .sum()
    )

    test_missing = int(
        test_dataframe
        .isnull()
        .sum()
        .sum()
    )

    train_duplicates = int(
        train_dataframe[
            "image_path"
        ]
        .duplicated()
        .sum()
    )

    test_duplicates = int(
        test_dataframe[
            "image_path"
        ]
        .duplicated()
        .sum()
    )

    train_classes = sorted(
        train_dataframe[
            "class_name"
        ]
        .unique()
        .tolist()
    )

    test_classes = sorted(
        test_dataframe[
            "class_name"
        ]
        .unique()
        .tolist()
    )

    if train_classes != test_classes:
        raise ValueError(
            "Training and testing directories "
            "contain different classes."
        )

    if train_missing > 0 or test_missing > 0:
        raise ValueError(
            "Missing values were found in "
            "the dataset records."
        )

    print(
        "Training duplicate paths:",
        train_duplicates
    )

    print(
        "Testing duplicate paths:",
        test_duplicates
    )

    return train_classes


def create_data_splits(
    train_full_dataframe,
    test_dataframe
):
    train_dataframe, validation_dataframe = (
        train_test_split(
            train_full_dataframe,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_SEED,
            stratify=train_full_dataframe[
                "class_name"
            ]
        )
    )

    train_dataframe = (
        train_dataframe
        .reset_index(drop=True)
    )

    validation_dataframe = (
        validation_dataframe
        .reset_index(drop=True)
    )

    test_dataframe = (
        test_dataframe
        .reset_index(drop=True)
    )

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe
    )


def get_train_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                degrees=10
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
    )


def get_evaluation_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
    )


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        dataframe,
        class_to_index,
        transform=None,
        return_path=False
    ):
        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.class_to_index = (
            class_to_index
        )

        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = row["image_path"]
        class_name = row["class_name"]

        try:
            with Image.open(
                image_path
            ) as image:
                image = image.convert("RGB")

        except Exception as error:
            raise RuntimeError(
                f"Unable to load image: "
                f"{image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_index[
            class_name
        ]

        if self.return_path:
            return image, label, image_path

        return image, label