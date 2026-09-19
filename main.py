import argparse
import json

from src.predict import (
    load_model,
    predict_image
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Brain Tumor MRI Classification"
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help=(
            "Path to the MRI image"
        )
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help=(
            "Confidence threshold for "
            "manual review"
        )
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if not (
        0.0
        <= arguments.confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "Confidence threshold must be "
            "between 0 and 1."
        )

    model, class_names, device = (
        load_model()
    )

    prediction = predict_image(
        image_path=arguments.image,
        model=model,
        class_names=class_names,
        device=device,
        confidence_threshold=(
            arguments.confidence_threshold
        )
    )

    print(
        json.dumps(
            prediction,
            indent=4
        )
    )


if __name__ == "__main__":
    main()
