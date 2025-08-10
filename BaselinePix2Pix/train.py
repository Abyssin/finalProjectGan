import tensorflow as tf
import os
import time
import datetime
import matplotlib.pyplot as plt
from IPython import display

from models import Generator, Discriminator
from losses import generator_loss, discriminator_loss, calculate_metrics
from data_processing import create_dataset_from_processed_data


def generate_images(model, test_input, tar, save_path=None):
    """Generate and display images during training"""
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot
        if i == 0:  # Input image is in [-1, 1]
            plt.imshow(display_list[i][:, :, 0] * 0.5 + 0.5, cmap='gray')
        else:  # Ground truth and prediction are in [0, 1]
            plt.imshow(display_list[i][:, :, 0], cmap='gray')
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


@tf.function
def train_step(input_image, target, generator, discriminator, 
               generator_optimizer, discriminator_optimizer, step, summary_writer):
    """Single training step"""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_dice_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_dice_loss', gen_dice_loss, step=step//1000)  # Changed from L1 to dice
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

    return gen_total_loss, gen_gan_loss, gen_dice_loss, disc_loss


def validate(generator, val_dataset, step, summary_writer):
    """Validation step"""
    val_gen_losses = []
    val_dice_losses = []
    val_metrics = {'iou': [], 'dice': [], 'pixel_accuracy': []}
    
    for input_image, target in val_dataset.take(10):  # Use subset for validation
        gen_output = generator(input_image, training=False)
        
        # Calculate losses (without discriminator for validation)
        gen_total_loss, gen_gan_loss, gen_dice_loss = generator_loss(
            tf.ones_like(tf.zeros([1, 30, 30, 1])),  # Dummy disc output for loss calc
            gen_output, target)
        
        val_gen_losses.append(gen_total_loss.numpy())
        val_dice_losses.append(gen_dice_loss.numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(target, gen_output)
        for key in val_metrics:
            val_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {key: sum(values) / len(values) for key, values in val_metrics.items()}
    avg_gen_loss = sum(val_gen_losses) / len(val_gen_losses)
    avg_dice_loss = sum(val_dice_losses) / len(val_dice_losses)
    
    # Log to tensorboard
    with summary_writer.as_default():
        tf.summary.scalar('val_gen_loss', avg_gen_loss, step=step//1000)
        tf.summary.scalar('val_dice_loss', avg_dice_loss, step=step//1000)
        tf.summary.scalar('val_iou', avg_metrics['iou'], step=step//1000)
        tf.summary.scalar('val_dice_score', avg_metrics['dice'], step=step//1000)
        tf.summary.scalar('val_pixel_accuracy', avg_metrics['pixel_accuracy'], step=step//1000)
    
    return avg_gen_loss, avg_metrics


def fit(train_ds, val_ds, test_ds, steps, generator, discriminator, 
        generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix, 
        summary_writer, output_dir):
    """Training loop"""
    # Use val set for visualization during training (following original pix2pix approach)
    example_input, example_target = next(iter(val_ds.take(1)))
    start = time.time()
    
    best_val_loss = float('inf')

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            # Generate sample images using val set (following original pix2pix approach)
            save_path = os.path.join(output_dir, f'sample_step_{step//1000}k.png')
            generate_images(generator, example_input, example_target, save_path)
            print(f"Step: {step//1000}k")

        # Training step
        gen_loss, gen_gan_loss, gen_dice_loss, disc_loss = train_step(
            input_image, target, generator, discriminator, 
            generator_optimizer, discriminator_optimizer, step, summary_writer)

        if (step+1) % 10 == 0:
            print('.', end='', flush=True)

        # Validation every 1000 steps
        if (step) % 1000 == 0 and step > 0:
            val_loss, val_metrics = validate(generator, val_ds, step, summary_writer)
            print(f"\nValidation - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint.save(file_prefix=os.path.join(output_dir, "best_model"))
                print(f"New best model saved! Val loss: {val_loss:.4f}")

        # Save checkpoint every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def main():
    """Main training function"""
    # Configuration
    DATA_DIR = "/workspaces/testGan/finalProjectGan/processed_data"
    OUTPUT_DIR = "/workspaces/testGan/finalProjectGan/BaselinePix2Pix/outputs"
    EPOCHS = 200
    STEPS_PER_EPOCH = 60  # Based on your 60 training samples
    TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH  # 12000 steps
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=== BASELINE PIX2PIX TRAINING ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total training steps: {TOTAL_STEPS}")
    print()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dataset_from_processed_data(DATA_DIR, 'train')
    val_dataset = create_dataset_from_processed_data(DATA_DIR, 'val')  # Used as "test" for visualization during training
    test_dataset = create_dataset_from_processed_data(DATA_DIR, 'test')  # True holdout for final comparison
    
    # Create models
    print("Creating models...")
    generator = Generator()
    discriminator = Discriminator()
    
    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}")

    # Create optimizers (same as original pix2pix)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Setup checkpoints
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'training_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

    # Setup logging
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    summary_writer = tf.summary.create_file_writer(
        log_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    print("Starting training...")
    print("Key differences from your current GAN:")
    print("- Uses RESIZE for val data instead of center crop (test set unchanged for fair comparison)")
    print("- Uses TensorFlow implementation instead of PyTorch")
    print("- Uses dice loss instead of L1 loss")
    print("- Uses sigmoid output activation instead of tanh")  
    print("- Uses 1-channel input/output instead of 3-channel")
    print("- Uses val set for visualization during training (following original pix2pix)")
    print("- Keeps test set as true holdout for final performance comparison")
    print()

    # Start training
    fit(train_dataset, val_dataset, test_dataset, TOTAL_STEPS, 
        generator, discriminator, generator_optimizer, discriminator_optimizer, 
        checkpoint, checkpoint_prefix, summary_writer, OUTPUT_DIR)

    print("\nTraining completed!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Best model: {OUTPUT_DIR}/best_model")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    # Set environment variables to completely disable XLA and JIT compilation
    import os
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir= --xla_disable_hlo_passes=all'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_DISABLE_XLA_JIT'] = '1'
    os.environ['TF_DISABLE_MKL'] = '1'
    
    
    # Completely disable XLA compilation and JIT
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({
        'disable_meta_optimizer': True,
        'disable_model_pruning': True,
        'disable_mlir_bridge': True,
        'auto_mixed_precision': False,
        'layout_optimizer': False,
        'constant_folding': False,
        'arithmetic_optimization': False,
        'dependency_optimization': False,
        'loop_optimization': False,
        'function_optimization': False,
        'debug_stripper': False,
        'scoped_allocator_optimization': False,
        'pin_to_host_optimization': False,
        'implementation_selector': False,
        'auto_parallel': {
            'enable': False
        }
    })
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Enable GPU memory growth to avoid OOM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    main()