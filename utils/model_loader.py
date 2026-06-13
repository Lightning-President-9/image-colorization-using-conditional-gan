from tensorflow.keras.models import load_model

from utils.loss_functions import (
    generator_loss_v1,
    generator_loss_v2
)

# ==========================================================
# MODEL CACHE
# ==========================================================

MODEL_CACHE = {}


def get_loss_function(loss_version):
    """
    Return selected loss function.
    """

    if loss_version == "v1":
        return generator_loss_v1

    return generator_loss_v2


def get_model(model_path, loss_version):
    """
    Load model once and reuse it.

    Cache key example:
    models/general/generator_model.keras_v1
    """

    cache_key = f"{model_path}_{loss_version}"

    if cache_key not in MODEL_CACHE:

        print(f"\nLoading model: {model_path}")
        print(f"Loss version: {loss_version}")

        loss_func = get_loss_function(
            loss_version
        )

        model = load_model(
            model_path,
            custom_objects={
                "generator_loss": loss_func,
                "generator_loss_v1": generator_loss_v1,
                "generator_loss_v2": generator_loss_v2
            },
            compile=False
        )

        MODEL_CACHE[cache_key] = model

        print("Model loaded successfully.\n")

    return MODEL_CACHE[cache_key]


def clear_model_cache():
    """
    Optional utility for debugging.
    """

    MODEL_CACHE.clear()

    print("Model cache cleared.")