import tensorflow as tf
from scripts.models import *
from scripts.utils import *

download_dataset()

ct_train_dataset = create_dataset('data/Dataset/images/train/CT', channels = 1, target_size = (360,360)).batch(1)
mri_train_dataset = create_dataset('data/Dataset/images/train/MRI', channels = 3 ,target_size = (360,360)).batch(1)
ct_iter = next(iter(ct_train_dataset))
mri_iter = next(iter(mri_train_dataset))
ct_test_dataset = create_dataset('data/Dataset/images/test/CT', channels = 1, target_size = (360,360)).batch(1)
mri_test_dataset = create_dataset('data/Dataset/images/test/MRI', channels = 3, target_size = (360,360)).batch(1)
test_ct = next(iter(ct_test_dataset))
test_mri = next(iter(mri_test_dataset))


if len(tf.config.list_physical_devices()) >= 3:
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices", strategy.num_replicas_in_sync)    
else: 
    strategy = tf.distribute.get_strategy()

initial_learning_rate = 0.001
decay_steps = 100000
decay_rate = 0.96

with strategy.scope():
    # Learning rate schedules for each optimizer
    ct_gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=True)

    mri_gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=True)

    ct_disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=True)

    mri_disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=True)
    
    ct_gen_opt = tf.keras.optimizers.Adam(learning_rate = ct_gen_lr_schedule, beta_1=0.5)
    mri_gen_opt = tf.keras.optimizers.Adam(learning_rate = mri_gen_lr_schedule, beta_1=0.5)
    ct_disc_opt = tf.keras.optimizers.Adam(learning_rate = ct_disc_lr_schedule, beta_1=0.5)
    mri_disc_opt = tf.keras.optimizers.Adam(learning_rate = mri_disc_lr_schedule, beta_1=0.5)

    # Compile the model
    cycle_gan = CycleGAN()
    cycle_gan.compile(ct_gen_opt, mri_gen_opt, ct_disc_opt, mri_disc_opt)

# Callbacks
callback = callbacks()
### Checkpoints
checkpoint_dir = './models/model_v1'

cycle_gan.fit(tf.data.Dataset.zip((ct_train_dataset, mri_train_dataset)), epochs=5, verbose = 1, callbacks = callback)