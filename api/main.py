import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
from PIL import UnidentifiedImageError

from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse
)

from src.config import (
    ALLOWED_IMAGE_TYPES,
    APP_NAME,
    APP_VERSION,
    CORS_ORIGINS,
    ENVIRONMENT,
    FINAL_MODEL_PATH,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB
)

from src.predict import (
    load_model,
    predict_pil_image
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "brain-tumor-api"
)


ml_resources = {
    "model": None,
    "class_names": None,
    "device": None,
    "metadata": None
}


@asynccontextmanager
async def lifespan(app):
    logger.info(
        "Loading model checkpoint from %s",
        FINAL_MODEL_PATH
    )

    (
        model,
        class_names,
        device,
        model_metadata
    ) = load_model(
        checkpoint_path=FINAL_MODEL_PATH
    )

    ml_resources["model"] = model
    ml_resources["class_names"] = (
        class_names
    )
    ml_resources["device"] = device
    ml_resources["metadata"] = (
        model_metadata
    )

    logger.info(
        "Model loaded successfully"
    )

    logger.info(
        "Inference device: %s",
        device
    )

    logger.info(
        "Classes: %s",
        class_names
    )

    yield

    logger.info(
        "Releasing model resources"
    )

    ml_resources.clear()


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Four-class brain MRI image "
        "classification API built with "
        "PyTorch and FastAPI. "
        "Educational demonstration only."
    ),
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],
    allow_headers=[
        "Content-Type",
        "Accept"
    ]
)


@app.get(
    "/",
    tags=["General"]
)
def root():
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "disclaimer": (
            "Educational demonstration only. "
            "Not intended for medical diagnosis."
        )
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"]
)
def health():
    model_loaded = (
        ml_resources.get("model")
        is not None
    )

    device = ml_resources.get(
        "device"
    )

    return {
        "status": (
            "healthy"
            if model_loaded
            else "unavailable"
        ),
        "model_loaded": model_loaded,
        "device": (
            str(device)
            if device is not None
            else "unavailable"
        ),
        "environment": ENVIRONMENT
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"]
)
def model_info():
    metadata = ml_resources.get(
        "metadata"
    )

    device = ml_resources.get(
        "device"
    )

    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    return {
        "model_name": metadata[
            "model_name"
        ],
        "model_version": metadata[
            "model_version"
        ],
        "framework": metadata[
            "framework"
        ],
        "classes": metadata[
            "classes"
        ],
        "image_size": metadata[
            "image_size"
        ],
        "device": str(device),
        "best_epoch": metadata[
            "best_epoch"
        ],
        "best_validation_loss": (
            metadata[
                "best_validation_loss"
            ]
        ),
        "best_validation_macro_f1": (
            metadata[
                "best_validation_macro_f1"
            ]
        ),
        "medical_diagnostic_use": False
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.70)
):
    model = ml_resources.get(
        "model"
    )

    class_names = ml_resources.get(
        "class_names"
    )

    device = ml_resources.get(
        "device"
    )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    if not (
        0.0
        <= confidence_threshold
        <= 1.0
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "confidence_threshold must be "
                "between 0 and 1."
            )
        )

    if file.content_type not in (
        ALLOWED_IMAGE_TYPES
    ):
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image type. "
                "Upload JPG, PNG, or WEBP."
            )
        )

    try:
        file_bytes = await file.read(
            MAX_UPLOAD_SIZE_BYTES + 1
        )

    finally:
        await file.close()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is empty."
            )
        )

    if (
        len(file_bytes)
        > MAX_UPLOAD_SIZE_BYTES
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                f"The uploaded image exceeds "
                f"{MAX_UPLOAD_SIZE_MB} MB."
            )
        )

    try:
        with Image.open(
            io.BytesIO(file_bytes)
        ) as uploaded_image:
            uploaded_image.verify()

        with Image.open(
            io.BytesIO(file_bytes)
        ) as uploaded_image:
            image = uploaded_image.convert(
                "RGB"
            )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not "
                "a valid image."
            )
        ) from error

    prediction_start_time = (
        time.perf_counter()
    )

    try:
        result = predict_pil_image(
            image=image,
            model=model,
            class_names=class_names,
            device=device,
            confidence_threshold=(
                confidence_threshold
            )
        )

    except Exception as error:
        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The model could not process "
                "the uploaded image."
            )
        ) from error

    prediction_duration = (
        time.perf_counter()
        - prediction_start_time
    )

    logger.info(
        (
            "Prediction completed | "
            "file=%s | class=%s | "
            "confidence=%.4f | "
            "duration_ms=%.2f"
        ),
        file.filename,
        result["predicted_class"],
        result["confidence"],
        prediction_duration * 1000
    )

    return {
        "filename": (
            file.filename
            or "uploaded_image"
        ),
        "predicted_class": result[
            "predicted_class"
        ],
        "confidence": result[
            "confidence"
        ],
        "requires_review": result[
            "requires_review"
        ],
        "confidence_threshold": result[
            "confidence_threshold"
        ],
        "class_probabilities": result[
            "class_probabilities"
        ],
        "disclaimer": (
            "Educational demonstration only. "
            "Not intended for medical diagnosis."
        )
    }