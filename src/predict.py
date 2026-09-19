from pathlib import Path

from PIL import Image

import torch

from src.config import (
    CONFIDENCE_THRESHOLD,
    FINAL_MODEL_PATH,
    IMAGE_SIZE
)

from src.dataset import get_evaluation_transform
from src.model import BrainTumorCNN
from src.utils import get_device


def load_checkpoint_file(
    checkpoint_path,
    device
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: "
            f"{checkpoint_path.resolve()}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

    except Exception:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

    return checkpoint


def load_model(
    checkpoint_path=FINAL_MODEL_PATH
):
    device = get_device()

    checkpoint = load_checkpoint_file(
        checkpoint_path=checkpoint_path,
        device=device
    )

    class_to_index = checkpoint[
        "class_to_index"
    ]

    class_names = [
        class_name
        for class_name, class_index in sorted(
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

    model_metadata = {
        "model_name": "BrainTumorCNN",
        "model_version": "1.0.0",
        "framework": "PyTorch",
        "classes": class_names,
        "image_size": int(
            checkpoint.get(
                "image_size",
                IMAGE_SIZE
            )
        ),
        "dropout": dropout,
        "best_epoch": int(
            checkpoint.get(
                "best_epoch",
                0
            )
        ),
        "best_validation_loss": float(
            checkpoint.get(
                "best_validation_loss",
                0.0
            )
        ),
        "best_validation_macro_f1": float(
            checkpoint.get(
                "best_validation_macro_f1",
                0.0
            )
        )
    }

    return (
        model,
        class_names,
        device,
        model_metadata
    )


def predict_pil_image(
    image,
    model,
    class_names,
    device,
    confidence_threshold=CONFIDENCE_THRESHOLD
):
    if not isinstance(
        image,
        Image.Image
    ):
        raise TypeError(
            "Input must be a PIL Image."
        )

    if not (
        0.0
        <= confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "Confidence threshold must be "
            "between 0 and 1."
        )

    image = image.convert("RGB")

    transform = get_evaluation_transform()

    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    with torch.inference_mode():
        logits = model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    confidence, predicted_index = torch.max(
        probabilities,
        dim=0
    )

    predicted_index = int(
        predicted_index.item()
    )

    confidence = float(
        confidence.item()
    )

    class_probabilities = {
        class_name: round(
            float(
                probabilities[
                    class_index
                ].item()
            ),
            6
        )
        for class_index, class_name in enumerate(
            class_names
        )
    }

    sorted_probabilities = dict(
        sorted(
            class_probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "predicted_class": class_names[
            predicted_index
        ],
        "confidence": round(
            confidence,
            6
        ),
        "requires_review": (
            confidence
            < confidence_threshold
        ),
        "confidence_threshold": round(
            confidence_threshold,
            4
        ),
        "class_probabilities": (
            sorted_probabilities
        )
    }


def predict_image(
    image_path,
    model,
    class_names,
    device,
    confidence_threshold=CONFIDENCE_THRESHOLD
):
    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path.resolve()}"
        )

    try:
        with Image.open(
            image_path
        ) as image:
            image = image.convert("RGB")

    except Exception as error:
        raise ValueError(
            f"Unable to process image: "
            f"{image_path.resolve()}"
        ) from error

    result = predict_pil_image(
        image=image,
        model=model,
        class_names=class_names,
        device=device,
        confidence_threshold=(
            confidence_threshold
        )
    )

    result["image_path"] = str(
        image_path.resolve()
    )

    result["disclaimer"] = (
        "Educational demonstration only. "
        "Not intended for medical diagnosis."
    )

    return result