import tensorflow as tf


LAMBDA = 100


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss for binary segmentation - TensorFlow implementation.
    
    Args:
        y_true: Ground truth masks [batch, H, W, 1] in range [0, 1]
        y_pred: Predicted masks [batch, H, W, 1] in range [0, 1] 
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        tf.Tensor: Dice loss value (1 - dice coefficient)
    """
    # Flatten tensors
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])  # [batch, H*W]
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])  # [batch, H*W]
    
    # Calculate intersection and dice coefficient
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)  # [batch]
    dice_coeff = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1) + smooth
    )
    
    # Dice loss = 1 - mean dice coefficient
    dice_loss_val = 1.0 - tf.reduce_mean(dice_coeff)
    
    return dice_loss_val


def generator_loss(disc_generated_output, gen_output, target):
    """
    MODIFIED pix2pix generator loss with dice loss instead of L1 loss.
    
    Changes from original:
    - Replaced L1 loss with dice loss for binary mask segmentation
    - Kept adversarial loss component unchanged
    """
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Adversarial loss (unchanged from original)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # CHANGE: Dice loss instead of L1 loss for binary mask segmentation
    dice_loss_val = dice_loss(target, gen_output)

    # Total generator loss: adversarial + lambda * dice
    total_gen_loss = gan_loss + (LAMBDA * dice_loss_val)

    return total_gen_loss, gan_loss, dice_loss_val


def discriminator_loss(disc_real_output, disc_generated_output):
    """Discriminator loss - unchanged from original TensorFlow implementation"""
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate segmentation metrics for evaluation - TensorFlow implementation.
    
    Args:
        y_true: Ground truth masks [batch, H, W, 1] in range [0, 1]
        y_pred: Predicted masks [batch, H, W, 1] in range [0, 1]
        threshold: Threshold for binarizing predictions
        
    Returns:
        dict: Metrics including IoU, Dice, pixel accuracy
    """
    # Binarize predictions
    pred_binary = tf.cast(y_pred > threshold, tf.float32)
    true_binary = tf.cast(y_true > threshold, tf.float32)
    
    # Flatten for computation
    pred_flat = tf.reshape(pred_binary, [tf.shape(pred_binary)[0], -1])  # [batch, H*W]
    true_flat = tf.reshape(true_binary, [tf.shape(true_binary)[0], -1])  # [batch, H*W]
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(pred_flat * true_flat, axis=1)  # [batch]
    union = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) - intersection
    
    # IoU
    iou = tf.reduce_mean((intersection + 1e-6) / (union + 1e-6))
    
    # Dice coefficient
    dice = tf.reduce_mean((2.0 * intersection + 1e-6) / 
                         (tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-6))
    
    # Pixel accuracy
    correct = tf.reduce_sum(tf.cast(pred_flat == true_flat, tf.float32), axis=1)
    total = tf.cast(tf.shape(pred_flat)[1], tf.float32)
    pixel_accuracy = tf.reduce_mean(correct / total)
    
    return {
        'iou': float(iou.numpy()),
        'dice': float(dice.numpy()),
        'pixel_accuracy': float(pixel_accuracy.numpy())
    }