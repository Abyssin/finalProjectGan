import tensorflow as tf


LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
    """Original TensorFlow pix2pix generator loss with L1 loss"""
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error (L1 loss)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    """Original TensorFlow pix2pix discriminator loss"""
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss