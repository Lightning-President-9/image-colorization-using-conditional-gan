import os
import time
import uuid
import zipfile
import threading
import logging

import numpy as np
from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file
)

from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from PIL import (
    Image,
    UnidentifiedImageError
)
from PIL.Image import DecompressionBombError

from utils.model_loader import get_model
from utils.image_utils import (
    prepare_image,
    save_comparison_image
)

load_dotenv()

# ==========================================================
# SECURITY CONFIG
# ==========================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp"
}

MAX_UPLOAD_MB = 10
MAX_WIDTH = 5000
MAX_HEIGHT = 5000

Image.MAX_IMAGE_PIXELS = 25_000_000

# ==========================================================
# APP CONFIG
# ==========================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

limiter = Limiter(
    get_remote_address,
    storage_uri=os.environ["REDIS_URL"],
    app=app,
    default_limits=["100/day"]

)

logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/generated"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

FILE_EXPIRY_SECONDS = 60

# ==========================================================
# MODEL PATHS
# ==========================================================

GENERAL_MODELS = {
    "generator_model":"models/general/generator_model.keras",
    "generator_model_shuffle":"models/general/generator_model_shuffle.keras",
    "generator_model_unet":"models/general/generator_model_unet.keras",
    "generator_model_unet_shuffle":"models/general/generator_model_unet_shuffle.keras",
    "generator_model_cgan":"models/general/generator_model_cgan.keras",
    "generator_model_cgan_shuffle":"models/general/generator_model_cgan_shuffle.keras",
}

CATEGORY_MODELS = {
    "airplane":"models/category/generator_model_cgan_airplane.keras",
    "car":"models/category/generator_model_cgan_car.keras",
    "car_v2":"models/category/generator_model_cgan_car_v2.keras",
    "cat":"models/category/generator_model_cgan_cat.keras",
    "dog":"models/category/generator_model_cgan_dog.keras",
    "flower":"models/category/generator_model_cgan_flower.keras",
    "fruit":"models/category/generator_model_cgan_fruit.keras",
    "landscape":"models/category/generator_model_cgan_landscape.keras",
    "motorbike":"models/category/generator_model_cgan_motorbike.keras",
    "person":"models/category/generator_model_cgan_person.keras"
}

# ==========================================================
# HELPERS
# ==========================================================

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

def safe_delete(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logging.error(str(e))

def delete_after_delay(file_paths, delay=FILE_EXPIRY_SECONDS):

    def cleanup():
        time.sleep(delay)

        for path in file_paths:
            safe_delete(path)

    thread = threading.Thread(
        target=cleanup,
        daemon=True
    )
    thread.start()

def validate_image(image_file):

    if not allowed_file(image_file.filename):
        raise ValueError("Unsupported file extension")

    if image_file.mimetype not in ALLOWED_MIME_TYPES:
        raise ValueError("Invalid MIME type")

    try:
        img = Image.open(image_file)
        img.verify()
    except Exception:
        raise ValueError("Invalid image file")

    image_file.seek(0)

    img = Image.open(image_file)

    if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
        raise ValueError(
            f"Maximum dimensions are "
            f"{MAX_WIDTH}x{MAX_HEIGHT}"
        )

    image_file.seek(0)

def safe_file_path(path):

    path = path.lstrip("/")
    abs_path = os.path.abspath(path)

    allowed_dirs = [
        os.path.abspath(UPLOAD_FOLDER),
        os.path.abspath(OUTPUT_FOLDER)
    ]

    if not any(
        abs_path.startswith(d)
        for d in allowed_dirs
    ):
        raise ValueError("Invalid file path")

    return abs_path

# ==========================================================
# ERROR HANDLERS
# ==========================================================

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):

    return jsonify({
        "success": False,
        "error": f"Maximum upload size is {MAX_UPLOAD_MB}MB"
    }), 413

# ==========================================================
# ROUTES
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        general_models=list(GENERAL_MODELS.keys()),
        category_models=list(CATEGORY_MODELS.keys())
    )

@app.route("/generate", methods=["POST"])
@limiter.limit("10/minute")
def generate():

    try:

        start_time = time.time()

        required_form = [
            "model_type",
            "selected_model",
            "loss_version"
        ]

        for field in required_form:
            if field not in request.form:
                return jsonify({
                    "success": False,
                    "error": f"Missing field: {field}"
                }), 400

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "Image not uploaded"
            }), 400

        image_file = request.files["image"]

        validate_image(image_file)

        model_type = request.form["model_type"]
        selected_model = request.form["selected_model"]
        loss_version = request.form["loss_version"]

        file_id = str(uuid.uuid4())

        original_filename = (
            f"{file_id}_"
            f"{secure_filename(image_file.filename)}"
        )

        original_path = os.path.join(
            UPLOAD_FOLDER,
            original_filename
        )

        image_file.save(original_path)

        (
            original_img,
            grayscale_img,
            model_input
        ) = prepare_image(original_path)

        grayscale_filename = (
            f"{file_id}_grayscale.png"
        )

        grayscale_path = os.path.join(
            OUTPUT_FOLDER,
            grayscale_filename
        )

        grayscale_img.save(grayscale_path)

        if model_type == "general":

            if selected_model not in GENERAL_MODELS:
                raise ValueError("Invalid model selected")

            model_path = GENERAL_MODELS[selected_model]

        else:

            if selected_model not in CATEGORY_MODELS:
                raise ValueError("Invalid model selected")

            model_path = CATEGORY_MODELS[selected_model]

        generator = get_model(
            model_path,
            loss_version
        )

        prediction = generator.predict(
            model_input,
            verbose=0
        )

        if prediction is None:
            raise ValueError("Model returned None")

        if prediction.size == 0:
            raise ValueError("Empty prediction")

        if np.isnan(prediction).any():
            raise ValueError("Prediction contains NaN")

        if np.isinf(prediction).any():
            raise ValueError("Prediction contains Infinity")

        prediction = np.clip(
            prediction[0],
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
            original_img.size
        )

        generated_filename = (
            f"{file_id}_generated.png"
        )

        generated_path = os.path.join(
            OUTPUT_FOLDER,
            generated_filename
        )

        generated_img.save(
            generated_path
        )

        comparison_filename = (
            f"{file_id}_comparison.png"
        )

        comparison_path = os.path.join(
            OUTPUT_FOLDER,
            comparison_filename
        )

        save_comparison_image(
            original_img,
            grayscale_img,
            generated_img,
            comparison_path
        )

        elapsed = round(
            time.time() - start_time,
            2
        )

        delete_after_delay([
            original_path,
            grayscale_path,
            generated_path,
            comparison_path
        ])

        return jsonify({
            "success": True,
            "original": "/" + original_path.replace("\\", "/"),
            "grayscale": "/" + grayscale_path.replace("\\", "/"),
            "generated": "/" + generated_path.replace("\\", "/"),
            "comparison": "/" + comparison_path.replace("\\", "/"),
            "inference_time": elapsed,
            "model": selected_model,
            "loss": loss_version
        })

    except (
        UnidentifiedImageError,
        DecompressionBombError
    ):

        return jsonify({
            "success": False,
            "error": "Invalid image"
        }), 400

    except Exception as e:

        logging.exception(e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/download_zip", methods=["POST"])
def download_zip():

    try:

        original = request.form["original"]
        grayscale = request.form["grayscale"]
        generated = request.form["generated"]
        comparison = request.form["comparison"]

        zip_name = (
            f"{uuid.uuid4()}.zip"
        )

        zip_path = os.path.join(
            OUTPUT_FOLDER,
            zip_name
        )

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zipf:

            files = [
                original,
                grayscale,
                generated,
                comparison
            ]

            for file_path in files:

                clean_path = safe_file_path(
                    file_path
                )

                zipf.write(
                    clean_path,
                    os.path.basename(clean_path)
                )

        delete_after_delay([zip_path])

        return send_file(
            zip_path,
            as_attachment=True
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/service-status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "service": "Image Colorization using Conditional GAN Service API"
    })


if __name__ == "__main__":
    app.run()
