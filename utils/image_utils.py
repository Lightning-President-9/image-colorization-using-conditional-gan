import os
import numpy as np

from PIL import (
    Image,
    ImageDraw,
    ImageFont
)

from tensorflow.keras.preprocessing.image import (
    img_to_array
)

# ==========================================================
# CONFIG
# ==========================================================

TARGET_SIZE = (120, 120)

# ==========================================================
# IMAGE DETECTION
# ==========================================================

def is_grayscale(img):
    """
    Check if image is grayscale.

    Returns:
        True  -> grayscale
        False -> color image
    """

    if img.mode == "L":
        return True

    if img.mode == "RGB":

        rgb = np.array(img)

        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        return np.array_equal(r, g) and np.array_equal(g, b)

    return False


# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def prepare_image(image_path):
    """
    Load image.

    If image is color:
        Convert to grayscale.

    Returns:
        original_img
        grayscale_img
        model_input
    """

    img = Image.open(image_path)

    original_img = img.copy()

    if not is_grayscale(img):
        img = img.convert("L")

    grayscale_img = img.copy()

    resized_img = img.resize(TARGET_SIZE)

    img_array = img_to_array(
        resized_img
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return (
        original_img,
        grayscale_img,
        img_array
    )


# ==========================================================
# SAVE GENERATED IMAGE
# ==========================================================

def save_generated_image(
    prediction,
    original_size,
    save_path
):
    """
    Convert model output to image
    and save.
    """

    prediction = np.clip(
        prediction,
        0,
        1
    )

    prediction = (
        prediction * 255
    ).astype(np.uint8)

    generated_img = Image.fromarray(
        prediction
    )

    generated_img = generated_img.resize(
        original_size
    )

    generated_img.save(save_path)

    return generated_img


# ==========================================================
# CREATE COMPARISON IMAGE
# ==========================================================

def save_comparison_image(
    original_img,
    grayscale_img,
    generated_img,
    save_path
):
    """
    Creates:

    Original | Grayscale | Generated
    """

    width = max(
        original_img.width,
        grayscale_img.width,
        generated_img.width
    )

    height = max(
        original_img.height,
        grayscale_img.height,
        generated_img.height
    )

    original = original_img.resize(
        (width, height)
    )

    grayscale = grayscale_img.resize(
        (width, height)
    )

    generated = generated_img.resize(
        (width, height)
    )

    if grayscale.mode != "RGB":
        grayscale = grayscale.convert("RGB")

    comparison = Image.new(
        "RGB",
        (width * 3, height + 60),
        color=(255, 255, 255)
    )

    comparison.paste(
        original.convert("RGB"),
        (0, 60)
    )

    comparison.paste(
        grayscale,
        (width, 60)
    )

    comparison.paste(
        generated.convert("RGB"),
        (width * 2, 60)
    )

    draw = ImageDraw.Draw(
        comparison
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            24
        )
    except:
        font = ImageFont.load_default()

    draw.text(
        (width // 2 - 40, 15),
        "Original",
        fill="black",
        font=font
    )

    draw.text(
        (width + width // 2 - 50, 15),
        "Grayscale",
        fill="black",
        font=font
    )

    draw.text(
        (width * 2 + width // 2 - 45, 15),
        "Generated",
        fill="black",
        font=font
    )

    comparison.save(save_path)

    return comparison


# ==========================================================
# CREATE ZIP FRIENDLY FILENAMES
# ==========================================================

def build_output_paths(
    output_folder,
    timestamp
):
    """
    Returns all generated paths.
    """

    return {
        "grayscale":
            os.path.join(
                output_folder,
                f"{timestamp}_grayscale.png"
            ),

        "generated":
            os.path.join(
                output_folder,
                f"{timestamp}_generated.png"
            ),

        "comparison":
            os.path.join(
                output_folder,
                f"{timestamp}_comparison.png"
            )
    }


# ==========================================================
# NORMALIZE OUTPUT
# ==========================================================

def prediction_to_image(
    prediction,
    original_size
):
    """
    Convert model prediction
    to PIL image.
    """

    prediction = np.clip(
        prediction,
        0,
        1
    )

    prediction = (
        prediction * 255
    ).astype(np.uint8)

    image = Image.fromarray(
        prediction
    )

    image = image.resize(
        original_size
    )

    return image


# ==========================================================
# GET IMAGE SIZE
# ==========================================================

def get_image_size(
    image_path
):
    img = Image.open(image_path)

    return img.size