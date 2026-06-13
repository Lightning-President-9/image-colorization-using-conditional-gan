import tensorflow as tf

from tensorflow.keras.losses import (
    BinaryCrossentropy,
    MeanSquaredError
)

from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable()
def generator_loss_v1(y_true, y_pred):
    """
    Binary Cross Entropy Loss

    Used when generator tries to fool
    discriminator into classifying
    generated images as real.
    """

    bce_loss = BinaryCrossentropy(
        from_logits=True
    )

    return bce_loss(
        K.ones_like(y_true),
        y_pred
    )


@tf.keras.utils.register_keras_serializable()
def generator_loss_v2(y_true, y_pred):
    """
    Mean Squared Error Loss

    Used for pixel-wise similarity
    between generated image and
    target image.
    """

    mse_loss = MeanSquaredError()

    return mse_loss(
        y_true,
        y_pred
    )


LOSS_FUNCTIONS = {
    "v1": generator_loss_v1,
    "v2": generator_loss_v2
}