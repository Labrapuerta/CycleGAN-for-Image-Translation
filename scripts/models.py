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

class Generator(keras.Model):
    def __init__(self, name = 'Generator',**kwargs):
        super(Generator, self).__init__(name = name, **kwargs)
        
        ### Downsampling layers
        self.downsample1 = Downsample(32, name = 'Donwsample_128')   ## (bs,128,128,32)
        self.downsample2 = Downsample(64, name = 'Donwsample_64')   ## (bs,64,64,64)
        self.downsample3 = Downsample(128, name = 'Donwsample_32')  ## (bs,32,32,128)
        self.downsample4 = Downsample(256, name = 'Donwsample_16')  ## (bs,16,16,256)
        self.downsample5 = Downsample(512, name = 'Donwsample_8')  ## (bs,8,8,512)
        self.downsample6 = Downsample(512, name = 'Donwsample_4')  ## (bs,4,4,512)
        self.downsample7 = Downsample(512, name = 'Donwsample_2')  ## (bs,2,2,512)
        self.downsample8 = Downsample(512, name = 'Donwsample_1')  ## (bs,1,1,512)   Goal 
        
        ## Upsampling layers
        self.upsample1 = Upsample(512, dropout = True, name = 'Upsample_2') ## (bs,2,2,512)
        self.upsample2 = Upsample(512, dropout = True, name = 'Upsample_4') ## (bs,4,4,512)
        self.upsample3 = Upsample(512,name = 'Upsample_8' ) ## (bs,8,8,512)
        self.upsample4 = Upsample(256, name = 'Upsample_16') ## (bs,16,16,256)
        self.upsample5 = Upsample(128, name = 'Upsample_32') ## (bs,32,32,128)
        self.upsample6 = Upsample(64, name = 'Upsample_64') ## (bs,64,64,64)
        self.upsample7 = Upsample(32, name = 'Upsample_128') ## (bs,128,128,32)
        self.final_upsample = Upsample(3, activation = 'sigmoid', name = 'Output_layer') # (bs,256,256,3)
        
        
    def call(self, inputs, training = False):
        ### Downsampling
        d1 = self.downsample1(inputs, training = training)
        d2 = self.downsample2(d1, training = training)
        d3 = self.downsample3(d2, training = training)
        d4 = self.downsample4(d3, training = training)
        d5 = self.downsample5(d4, training = training)
        d6 = self.downsample6(d5, training = training)
        d7 = self.downsample7(d6, training = training)
        d8 = self.downsample8(d7, training = training)
        ## Upsampling
        u1 = self.upsample1(d8, training = training)
        u1_concat = tf.concat([u1, d7], axis=-1)
        u2 = self.upsample2(u1_concat, training = training)
        u2_concat = tf.concat([u2, d6], axis=-1)
        u3 = self.upsample3(u2_concat, training = training)
        u3_concat = tf.concat([u3, d5], axis=-1)
        u4 = self.upsample4(u3_concat, training = training)
        u4_concat = tf.concat([u4, d4], axis=-1)
        u5 = self.upsample5(u4_concat, training = training)
        u5_concat = tf.concat([u5, d3], axis=-1)
        u6 = self.upsample6(u5_concat, training = training)
        u6_concat = tf.concat([u6, d2], axis=-1)
        u7 = self.upsample7(u6_concat, training = training)
        u7_concat = tf.concat([u7, d1], axis=-1)
        x = self.final_upsample(u7_concat, training = training)
        return x

class Discriminator(keras.Model):
    def __init__(self, name = 'Discriminator', dense = 64, activation = 'leaky_relu', **kwargs):
        super(Discriminator, self).__init__(name = name, **kwargs)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.downsample1 = Downsample(64, name = 'Downsample_128')   ## (bs,128,128,64)
        self.downsample2 = Downsample(128, name = 'Downsample_64')  ## (bs, 64,64,128)
        self.downsample3 = Downsample(256, name = 'Downsample_32')  ## (bs, 32,32,256)
        self.downsample4 = Downsample(256, name = 'Downsample_16')  ## (bs, 32,32,256)
        self.zeropad = layers.ZeroPadding2D(name = 'Zero_Padding')
        self.downsample5 = Downsample(512, strides = (1,1), kernel_size = (2,2), name = 'Downsample_stride_1')  ## (bs,32,32,512)
        self.flatten = layers.Flatten(name = 'Flatten_layer')
        self.dense = layers.Dense(dense, name = f'Hiden_layer_{dense}')
        self.dropout = layers.Dropout(0.2, name = 'Dropout')
        self.activation = layers.Activation(activation,name =  f'{activation}')
        self.last = layers.Dense(1, activation = 'sigmoid')
        
    def call(self,inputs, training = False):
        x = self.downsample1(inputs, training = training)
        x = self.downsample2(x, training = training)
        x = self.downsample3(x, training = training)
        x = self.downsample4(x, training = training)
        x = self.zeropad(x)
        x = self.downsample5(x, training =training)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.last(x)
        return x


class CycleGAN(keras.Model):
    def __init__(self, name = 'CycleGAN', lambda_cycle = 10, lambda_identity = 5):
        super(CycleGAN,self).__init__(name = name)
        self.monet_generator_ = Generator(name = 'Monet_Generator')
        self.monet_discriminator_ = Discriminator(name = 'Monet_Discriminator')
        self.photo_generator_ = Generator(name = 'Photo_Generator')
        self.photo_discriminator_ = Discriminator(name = 'Photo_Discriminator')
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self, m_gen_optimizer, p_gen_optimizer, m_disc_optimizer, p_disc_optimizer,  **kwargs):
        super(CycleGAN, self).compile(**kwargs)
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
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
        real_monet, real_photo = batch_data
        ### Persistent True to compute multiple gradients for the same tape
        with tf.GradientTape(persistent = True) as tape:
            
            ### Forward pass
            fake_monet = self.monet_generator_(real_photo, training = True) ## Photo to fake Monet
            cycled_photo =self.photo_generator_(fake_monet, training = True) ## fake Monet to photo
            fake_photo = self.photo_generator_(real_monet, training = True)  ## Monet to fake photo
            cycled_monet = self.monet_generator_(fake_photo, training = True) ## fake Photo to Monet
            
            ### Identity mapping (G(x) to x, F(y) to y)
            same_monet = self.monet_generator_(real_monet, training = True) ## Monet to monet
            same_photo = self.photo_generator_(real_photo, training = True) ## Photo to photo
            
            #### Discriminator prediction
            disc_real_monet = self.monet_discriminator_(real_monet, training = True) ## Monet is real Monet
            disc_fake_monet = self.monet_discriminator_(fake_monet, training = True) ## Fake Monet is real Monet
            disc_real_photo = self.photo_discriminator_(real_photo, training = True) ## Photo is a real photo
            disc_fake_photo = self.photo_discriminator_(fake_photo, training = True) ## Fake Photo is a real photo
            
            ### Compute the loss functions
            ## Generators
            gen_monet_loss = self.generator_loss_(disc_fake_monet)  
            gen_photo_loss = self.generator_loss_(disc_fake_photo)
            ## Discriminators
            disc_monet_loss = self.discriminator_loss_(disc_real_monet, disc_fake_monet)
            disc_photo_loss = self.discriminator_loss_(disc_real_photo, disc_fake_photo)
            ### Cycle
            cycle_loss = self.cycle_loss_(real_monet, cycled_monet) + self.cycle_loss_(real_photo, cycled_photo)
            ### Identity
            identity_loss = self.identity_loss_(real_monet,same_monet) + self.identity_loss_(real_photo, same_photo)
            ### Generators total loss
            total_gen_monet_loss = gen_monet_loss + cycle_loss + identity_loss
            total_gen_photo_loss = gen_photo_loss + cycle_loss + identity_loss
            
        ###Gradients
        monet_generator_gradients = tape.gradient(total_gen_monet_loss, self.monet_generator_.trainable_variables)
        photo_generator_gradients = tape.gradient(total_gen_photo_loss, self.photo_generator_.trainable_variables)
        monet_discriminator_gradients = tape.gradient(disc_monet_loss, self.monet_discriminator_.trainable_variables)
        photo_discriminator_gradients = tape.gradient(disc_photo_loss, self.photo_discriminator_.trainable_variables)

        ### Aplying gradients
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,self.monet_generator_.trainable_variables))
        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,self.photo_generator_.trainable_variables))
        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,self.monet_discriminator_.trainable_variables))
        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,self.photo_discriminator_.trainable_variables))

        return {
            "monet_gen_loss": total_gen_monet_loss,
            "photo_gen_loss": total_gen_photo_loss,
            "monet_disc_loss": disc_monet_loss,
            "photo_disc_loss": disc_photo_loss}