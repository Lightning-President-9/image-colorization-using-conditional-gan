# Image Colorization using Conditional GAN
## Overview

This project focuses on automatic image colorization using multiple deep learning architectures including:

* GAN (Generative Adversarial Network)
* GAN + Pixel Shuffle
* U-Net
* U-Net + Pixel Shuffle
* Conditional GAN (CGAN)
* CGAN + Pixel Shuffle

The system supports both:

1. **General Image Colorization Models**
2. **Category-Specific Colorization Models**

A Flask-based web application is provided for inference, allowing users to upload grayscale images and generate colorized outputs using trained models.

---

## Dataset Sources

### Landscape Dataset

* Total Available Images: **7129**
* Images Used: **1053**
* Source:
  https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization

### Flower Dataset

* Total Images: **210**
* Source:
  https://://[www.kaggle.com/datasets/olgabelitskaya/flower-color-images](http://www.kaggle.com/datasets/olgabelitskaya/flower-color-images)

### Natural Images Dataset

Source:
https://www.kaggle.com/datasets/prasunroy/natural-images

Images used for category-specific training:

| Category  | Images |
| --------- | -----: |
| Airplane  |    727 |
| Car       |    968 |
| Cat       |    885 |
| Dog       |    702 |
| Flower    |    843 |
| Fruit     |   1000 |
| Motorbike |    788 |
| Person    |    986 |

### Additional Car Dataset

* Images Used: **200**
* Used to create an enhanced car-specific model (Car V2)
* Source:
  https://www.kaggle.com/datasets/rohanraoeravelli/colour-greyscale-dataset

---

## Image Preprocessing

All images were preprocessed before training.

### Preprocessing Pipeline

1. Resize images to **120 × 120**
2. Convert RGB images to grayscale inputs
3. Normalize pixel values
4. Generate paired grayscale-color training samples

Final Input Shape:

```text
120 × 120 × 1
```

Final Output Shape:

```text
120 × 120 × 3
```

---

## Training Strategy

### General Models

The general models were trained using a mixed dataset containing images from multiple categories.

#### Sequential Training

For models without the shuffle suffix:

```text
generator_model
generator_model_unet
generator_model_cgan
```

Images were processed sequentially according to file naming order.

#### Shuffled Training

For models with the shuffle suffix:

```text
generator_model_shuffle
generator_model_unet_shuffle
generator_model_cgan_shuffle
```

Training data was randomly shuffled before each training process.

---

## Category-Specific Models

Separate CGAN models were trained for individual categories:

| Model                          | Category               |
| ------------------------------ | ---------------------- |
| generator_model_cgan_airplane  | Airplane               |
| generator_model_cgan_car       | Car                    |
| generator_model_cgan_car_v2    | Car (Enhanced Dataset) |
| generator_model_cgan_cat       | Cat                    |
| generator_model_cgan_dog       | Dog                    |
| generator_model_cgan_flower    | Flower                 |
| generator_model_cgan_fruit     | Fruit                  |
| generator_model_cgan_landscape | Landscape              |
| generator_model_cgan_motorbike | Motorbike              |
| generator_model_cgan_person    | Person                 |

These specialized models learn category-specific color distributions and object characteristics, often producing more realistic results for their respective domains.

---

## Web Application Features

### AI Colorization

* Upload grayscale images
* Select model architecture
* Generate colorized outputs
* Compare original, grayscale, and generated images

### Security Features

* Allowed file extension validation
* MIME type verification
* Upload size restriction
* Image dimension validation
* Decompression bomb protection
* Secure filename handling
* Path traversal protection
* Automatic file cleanup
* Request rate limiting

### Download Support

Generated outputs can be downloaded as a ZIP archive containing:

* Original image
* Grayscale image
* Generated image
* Comparison image

---

## Project Structure

```text
project/
│
├── app.py
├── models/
│   ├── general/
│   └── category/
│
├── static/
│   ├── uploads/
│   └── generated/
│
├── templates/
│   └── index.html
│
└── utils/
    ├── model_loader.py
    ├── image_utils.py
    └── loss_functions.py
```