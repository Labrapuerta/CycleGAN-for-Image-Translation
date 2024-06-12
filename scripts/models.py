import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class Downsample(keras.layers.Layer):
    def __init__(self,filters, kernel_size = (2,2), strides=(2, 2), padding='same', activation = 'leaky_relu', **kwargs):
        
        super(Downsample, self).__init__(**kwargs)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.conv = layers.Conv2D(filters, kernel_size = kernel_size, 
                                  strides = strides, padding = padding, kernel_initializer = self.initializer)
        self.activation = layers.Activation(activation = activation)
        self.gn = layers.GroupNormalization(groups = 1)

        
    def call(self, inputs,training = False):
        x = self.conv(inputs)
        x = self.activation(x)
        x = self.gn(x, training = training)
        return x
    
    def get_config(self):
        config = super(Downsample, self).get_config()
        config.update({
            'filters': self.conv.filters,
            'kernel_size': self.conv.kernel_size,
            'strides': self.conv.strides,
            'padding': self.conv.padding,
            'activation': self.activation.activation,
        })
        return config
    
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters = 512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation = 'leaky_relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn1 = layers.BatchNormalization()
        self.activation = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return layers.Add()([inputs, x])

class Upsample(keras.layers.Layer):
    def __init__(self,filters, kernel_size = (2,2), strides=(2, 2), padding='same', activation = 'leaky_relu', dropout = 0.2,**kwargs):
        
        super(Upsample, self).__init__(**kwargs)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv = layers.Conv2DTranspose(filters, kernel_size = kernel_size, 
                                           strides = strides, padding = padding, kernel_initializer = self.initializer)
        self.dropout = layers.Dropout(dropout)
        self.activation = layers.Activation(activation = activation)
        self.gn = layers.GroupNormalization(groups = 1)

    
    def call(self, inputs,training = False, dropout = False):
        x = self.conv(inputs)
        if dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.gn(x, training = training)
        return x
    
    def get_config(self):
        config = super(Upsample, self).get_config()
        config.update({
            'filters': self.conv.filters,
            'kernel_size': self.conv.kernel_size,
            'strides': self.conv.strides,
            'padding': self.conv.padding,
            'activation': self.activation.activation,
        })
        return config

class Generator(tf.keras.Model):
    def __init__(self, name='Generator', output_channels=3, **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        self.output_channels = output_channels
        
        # Downsampling layers
        self.downsample1 = Downsample(64, kernel_size=(3, 3), strides=(1, 1), name='Downsample_360')   # (bs, 360, 360, 64)
        self.downsample2 = Downsample(128, name='Downsample_180')                                      # (bs, 180, 180, 128)
        self.downsample3 = Downsample(256, name='Downsample_90')                                       # (bs, 90, 90, 256)
        self.downsample4 = Downsample(256, name='Downsample_30', strides=(3, 3))                       # (bs, 30, 30, 256)
        self.downsample5 = Downsample(512, name='Downsample_10', strides=(3, 3))                       # (bs, 10, 10, 512)
        
        # Residual Blocka
        self.residual1 = ResidualBlock()
        self.residual2 = ResidualBlock()
        self.residual3 = ResidualBlock()
        
        # Upsampling layers
        self.upsample1 = Upsample(256, name='Upsample_30', strides=(3, 3))                             # (bs, 30, 30, 256)
        self.upsample2 = Upsample(256, name='Upsample_90', strides=(3, 3))                             # (bs, 90, 90, 256)
        self.upsample3 = Upsample(128, name='Upsample_180')                                            # (bs, 180, 180, 128)
        self.upsample4 = Upsample(64, name='Upsample_360')                                             # (bs, 360, 360, 64)
        self.upsample5 = Upsample(self.output_channels, activation='tanh', kernel_size=(3, 3), strides=(1, 1), name='Final_layer')  # (bs, 360, 360, output_channels)
   
    def call(self, inputs, training=False):
        # Downsampling
        d1 = self.downsample1(inputs, training=training)
        d2 = self.downsample2(d1, training=training)
        d3 = self.downsample3(d2, training=training)
        d4 = self.downsample4(d3, training=training)
        d5 = self.downsample5(d4, training=training)
        
        ## Residual Blocks
        r1 = self.residual1(d5)
        r2 = self.residual2(r1)
        r3 = self.residual3(r2)

        # Upsampling
        u1 = self.upsample1(r3, training=training)
        u1_concat = tf.concat([u1, d4], axis=-1)
        u2 = self.upsample2(u1_concat, training=training)
        u2_concat = tf.concat([u2, d3], axis=-1)
        u3 = self.upsample3(u2_concat, training=training)
        u3_concat = tf.concat([u3, d2], axis=-1)
        u4 = self.upsample4(u3_concat, training=training)
        u4_concat = tf.concat([u4, d1], axis=-1)
        x = self.upsample5(u4_concat, training=training)
        return x
    
class Discriminator(tf.keras.Model):
    def __init__(self, name = 'Discriminator', dense = 64, activation = 'leaky_relu', **kwargs):
        super(Discriminator, self).__init__(name = name, **kwargs)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.downsample1 = Downsample(64, name = 'Downsample_128')   ## (bs,180,180,64)
        self.downsample2 = Downsample(128, name = 'Downsample_64')  ## (bs, 90,90,128)
        self.downsample3 = Downsample(256, name = 'Downsample_32')  ## (bs, 45,45,256)
        self.downsample4 = Downsample(256, name = 'Downsample_16', strides = (3,3))  ## (bs,15,15,256)
        self.zeropad = layers.ZeroPadding2D(name = 'Zero_Padding')
        self.downsample5 = Downsample(512, strides = (1,1), kernel_size = (2,2), name = 'Downsample_stride_1')  ## (bs,15,15,512)
        self.flatten = layers.Flatten(name = 'Flatten_layer')
        self.dense = layers.Dense(dense, name = f'Hiden_layer_{dense}')
        self.dropout = layers.Dropout(0.2, name = 'Dropout')
        self.activation = layers.Activation(activation,name =  f'{activation}')
        self.last = layers.Dense(1, activation = 'sigmoid')
        
    def call(self,inputs, training = False, dropout = False):
        x = self.downsample1(inputs, training = training)
        x = self.downsample2(x, training = training)
        x = self.downsample3(x, training = training)
        x = self.downsample4(x, training = training)
        x = self.zeropad(x)
        x = self.downsample5(x, training =training)
        x = self.flatten(x)
        x = self.dense(x)
        if dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.last(x)
        return x


class CycleGAN(tf.keras.Model):
    def __init__(self, name = 'CycleGAN', lambda_cycle = 10, lambda_identity = 5):
        super(CycleGAN,self).__init__(name = name)
        self.ct_generator_ = Generator(name = 'CT_Generator') #CT to MRI
        self.ct_discriminator_ = Discriminator(name = 'CT_Discriminator')
        self.mri_generator_ = Generator(name = 'MRI_Generator') # MRI to CT
        self.mri_discriminator_ = Discriminator(name = 'MRI_Discriminator')
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self, ct_gen_optimizer, mri_gen_optimizer, ct_disc_optimizer, mri_disc_optimizer,  **kwargs):
        super(CycleGAN, self).compile(**kwargs)
        self.ct_gen_optimizer = ct_gen_optimizer
        self.mri_gen_optimizer = mri_gen_optimizer
        self.ct_disc_optimizer = ct_disc_optimizer
        self.mri_disc_optimizer = mri_disc_optimizer
        self.bce_ = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        
    def generator_loss_(self, disc_output):
        return self.bce_(tf.ones_like(disc_output), disc_output)
    
    def discriminator_loss_(self, real_output, fake_output):
        real_loss = self.bce_(tf.ones_like(real_output), real_output)
        fake_loss = self.bce_(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    def cycle_loss_(self,real_image, cycled_image):
        return self.lambda_cycle * tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    def identity_loss_(self, real_image, same_image):
        return self.lambda_identity * tf.reduce_mean(tf.abs(real_image - same_image))
    
    def train_step(self, batch_data):
        real_ct, real_mri = batch_data
        ### Persistent True to compute multiple gradients for the same tape
        with tf.GradientTape(persistent = True) as tape:
            
            ### Forward pass
            fake_ct = self.ct_generator_(real_mri, training = True) ## MRI to fake CT
            cycled_ct =self.mri_generator_(fake_ct, training = True) ## fake CT to MRI
            fake_mri = self.mri_generator_(real_ct, training = True)  ## Monet to fake photo
            cycled_mri = self.ct_generator_(fake_mri, training = True) ## fake Photo to Monet
            
            ### Identity mapping (G(x) to x, F(y) to y)
            same_ct = self.ct_generator_(real_ct, training = True) ## Monet to monet
            same_mri = self.mri_generator_(real_mri, training = True) ## Photo to photo
            
            #### Discriminator prediction
            disc_real_ct = self.ct_discriminator_(real_ct, training = True) ## Monet is real Monet
            disc_fake_ct = self.ct_discriminator_(fake_ct, training = True) ## Fake Monet is real Monet
            disc_real_mri = self.mri_discriminator_(real_mri, training = True) ## Photo is a real photo
            disc_fake_mri = self.mri_discriminator_(fake_mri, training = True) ## Fake Photo is a real photo
            
            ### Compute the loss functions
            ## Generators
            gen_ct_loss = self.generator_loss_(disc_fake_ct)  
            gen_mri_loss = self.generator_loss_(disc_fake_mri)
            ## Discriminators
            disc_ct_loss = self.discriminator_loss_(disc_real_ct, disc_fake_ct)
            disc_mri_loss = self.discriminator_loss_(disc_real_mri, disc_fake_mri)
            ### Cycle
            cycle_loss = self.cycle_loss_(real_ct, cycled_ct) + self.cycle_loss_(real_mri, cycled_mri)
            ### Identity
            identity_loss = self.identity_loss_(real_ct,same_ct) + self.identity_loss_(real_mri, same_mri)
            ### Generators total loss
            total_gen_ct_loss = gen_ct_loss + cycle_loss + identity_loss
            total_gen_mri_loss = gen_mri_loss + cycle_loss + identity_loss
            
        ###Gradients
        ct_generator_gradients = tape.gradient(total_gen_ct_loss, self.ct_generator_.trainable_variables)
        mri_generator_gradients = tape.gradient(total_gen_mri_loss, self.mri_generator_.trainable_variables)
        ct_discriminator_gradients = tape.gradient(disc_ct_loss, self.ct_discriminator_.trainable_variables)
        mri_discriminator_gradients = tape.gradient(disc_mri_loss, self.mri_discriminator_.trainable_variables)

        ### Aplying gradients
        self.ct_gen_optimizer.apply_gradients(zip(ct_generator_gradients,self.ct_generator_.trainable_variables))
        self.mri_gen_optimizer.apply_gradients(zip(mri_generator_gradients,self.mri_generator_.trainable_variables))
        self.ct_disc_optimizer.apply_gradients(zip(ct_discriminator_gradients,self.ct_discriminator_.trainable_variables))
        self.mri_disc_optimizer.apply_gradients(zip(mri_discriminator_gradients,self.mri_discriminator_.trainable_variables))

        return {
            "ct_gen_loss": total_gen_ct_loss,
            "mri_gen_loss": total_gen_mri_loss,
            "ct_disc_loss": disc_ct_loss,
            "mri_disc_loss": disc_mri_loss,
            "cycle loss" : cycle_loss,
            "identity loss": identity_loss}