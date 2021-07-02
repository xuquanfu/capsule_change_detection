'''
Code for
1. Q. Xu, K. Chen, X. Sun, Y. Zhang, H. Li and G. Xu, "Pseudo-Siamese Capsule Network for Aerial Remote Sensing Images Change Detection," in IEEE Geoscience and Remote Sensing Letters, doi: 10.1109/LGRS.2020.3022512.
2. Change Capsule Network for Optical Remote Sensing Images Change Detection

If you have any questions, please email me at xuquanfu18@mails.ucas.ac.cn.

We sincerely Thank Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241) for the code of SegCaps.
'''

from keras import layers, models
from keras import backend as K
from keras import regularizers
from keras import initializers
import tensorflow as tf
    
K.set_image_data_format('channels_last')

from capsule_layers import ConvCapsuleLayer, \
    DeconvCapsuleLayer, Mask, Length, HalfLength, \
    OutCosLayer, MultiplyLayer, ReduceSumLayer, DivideLayer

#loss function, parameter can be adjusted with the data set
def margin_focal_loss(y_true, y_pred):
    return K.mean(-y_true * K.maximum(0.0,0.9 - y_pred) * K.log(y_pred) -  1.5*(1 - y_true) * K.maximum(0.0,y_pred - 0.1) * K.log(1. - y_pred))

#MSE
def mean_squared_error_for_rescon(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true[:,:,:,:-1])*tf.expand_dims(y_true[:,:,:,-1],axis=-1),axis=-1)





# Pseudo-Siamese Capsule Network
def SiameseCapsNetR3(input_shape, n_class=2):
    x1 = layers.Input(shape=input_shape) # x1=keras_shape(None, 112, 112, 3)
    x2 = layers.Input(shape=input_shape)  # x2=keras_shape(None, 112, 112, 3)

    # Layer 1: Just a conventional Conv2D layer
    #kernel_size=5
    conv1_1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1_1')(x1)
    conv2_1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv2_1')(x2)


    # Reshape layer to be 1 capsule x [filters] atoms
    _, H1, W1, C1= conv1_1.get_shape() # _, 512, 512, 32
    _, H2, W2, C2 = conv2_1.get_shape()  # _, 512, 512, 32
    # conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)
    conv1_1_reshaped = layers.Reshape((H1.value, W1.value, 1, C1.value))(conv1_1)
    conv2_1_reshaped = layers.Reshape((H2.value, W2.value, 1, C2.value))(conv2_1)


    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps1 = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps1')(conv1_1_reshaped)

    primary_caps2 = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps2')(conv2_1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap1_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='conv_cap1_2_1')(primary_caps1)

    conv_cap2_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                     routings=3, name='conv_cap2_2_1')(primary_caps2)

    # Layer 2: Convolutional Capsule
    conv_cap1_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap1_2_2')(conv_cap1_2_1)
    conv_cap2_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                     routings=3, name='conv_cap2_2_2')(conv_cap2_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap1_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap1_3_1')(conv_cap1_2_2)
    conv_cap2_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                     routings=3, name='conv_cap2_3_1')(conv_cap2_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap1_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap1_3_2')(conv_cap1_3_1)
    conv_cap2_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                     routings=3, name='conv_cap2_3_2')(conv_cap2_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap1_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap1_4_1')(conv_cap1_3_2)
    conv_cap2_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                     routings=3, name='conv_cap2_4_1')(conv_cap2_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap1_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap1_1_1')(conv_cap1_4_1)
    deconv_cap2_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                         scaling=2, padding='same', routings=3,
                                         name='deconv_cap2_1_1')(conv_cap2_4_1)

    # Skip connection
    up1_1 = layers.Concatenate(axis=-2, name='up1_1')([deconv_cap1_1_1, conv_cap1_3_1])
    up2_1 = layers.Concatenate(axis=-2, name='up2_1')([deconv_cap2_1_1, conv_cap2_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap1_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap1_1_2')(up1_1)
    deconv_cap2_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap2_1_2')(up2_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap1_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap1_2_1')(deconv_cap1_1_2)
    deconv_cap2_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                         scaling=2, padding='same', routings=3,
                                         name='deconv_cap2_2_1')(deconv_cap2_1_2)

    # Skip connection
    up1_2 = layers.Concatenate(axis=-2, name='up1_2')([deconv_cap1_2_1, conv_cap1_2_1])
    up2_2 = layers.Concatenate(axis=-2, name='up2_2')([deconv_cap2_2_1, conv_cap2_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap1_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap1_2_2')(up1_2)
    deconv_cap2_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap2_2_2')(up2_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap1_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap1_3_1')(deconv_cap1_2_2)
    deconv_cap2_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap2_3_1')(deconv_cap2_2_2)

    # Skip connection
    up1_3 = layers.Concatenate(axis=-2, name='up1_3')([deconv_cap1_3_1, conv1_1_reshaped])
    up2_3 = layers.Concatenate(axis=-2, name='up2_3')([deconv_cap2_3_1, conv2_1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps1 = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps1')(up1_3)
    seg_caps2 = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                 routings=3, name='seg_caps2')(up2_3)
    concatenate = layers.Concatenate(axis=-2, name='concatenate')([seg_caps1, seg_caps2])
    finally_seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='finally_seg_caps')(concatenate)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(finally_seg_caps)
    train_model=models.Model(inputs=[x1,x2], outputs=out_seg)
    return train_model




#Change Capsule Network for Change Detection
def ChangeCapsule(input_shape, n_class=2):
    x1 = layers.Input(shape=input_shape) # x1=keras_shape(None, 112, 112, 3)
    x2 = layers.Input(shape=input_shape)  # x2=keras_shape(None, 112, 112, 3)

    # Layer 1: Just a conventional Conv2D layer
    #kernel_size=5
    conv1_1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1_1')(x1)
    conv2_1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv2_1')(x2)


    # Reshape layer to be 1 capsule x [filters] atoms
    _, H1, W1, C1= conv1_1.get_shape() # _, 512, 512, 32
    _, H2, W2, C2 = conv2_1.get_shape()  # _, 512, 512, 32
    # conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)
    conv1_1_reshaped = layers.Reshape((H1.value, W1.value, 1, C1.value))(conv1_1)
    conv2_1_reshaped = layers.Reshape((H2.value, W2.value, 1, C2.value))(conv2_1)


    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps1 = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps1')(conv1_1_reshaped)

    primary_caps2 = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps2')(conv2_1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap1_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='conv_cap1_2_1')(primary_caps1)

    conv_cap2_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                     routings=3, name='conv_cap2_2_1')(primary_caps2)

    # Layer 2: Convolutional Capsule
    conv_cap1_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap1_2_2')(conv_cap1_2_1)
    conv_cap2_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                     routings=3, name='conv_cap2_2_2')(conv_cap2_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap1_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap1_3_1')(conv_cap1_2_2)
    conv_cap2_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                     routings=3, name='conv_cap2_3_1')(conv_cap2_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap1_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap1_3_2')(conv_cap1_3_1)
    conv_cap2_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                     routings=3, name='conv_cap2_3_2')(conv_cap2_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap1_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap1_4_1')(conv_cap1_3_2)
    conv_cap2_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                     routings=3, name='conv_cap2_4_1')(conv_cap2_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap1_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap1_1_1')(conv_cap1_4_1)
    deconv_cap2_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                         scaling=2, padding='same', routings=3,
                                         name='deconv_cap2_1_1')(conv_cap2_4_1)

    # Skip connection
    up1_1 = layers.Concatenate(axis=-2, name='up1_1')([deconv_cap1_1_1, conv_cap1_3_1])
    up2_1 = layers.Concatenate(axis=-2, name='up2_1')([deconv_cap2_1_1, conv_cap2_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap1_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap1_1_2')(up1_1)
    deconv_cap2_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap2_1_2')(up2_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap1_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap1_2_1')(deconv_cap1_1_2)
    deconv_cap2_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                         scaling=2, padding='same', routings=3,
                                         name='deconv_cap2_2_1')(deconv_cap2_1_2)

    # Skip connection
    up1_2 = layers.Concatenate(axis=-2, name='up1_2')([deconv_cap1_2_1, conv_cap1_2_1])
    up2_2 = layers.Concatenate(axis=-2, name='up2_2')([deconv_cap2_2_1, conv_cap2_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap1_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap1_2_2')(up1_2)
    deconv_cap2_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap2_2_2')(up2_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap1_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap1_3_1')(deconv_cap1_2_2)
    deconv_cap2_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap2_3_1')(deconv_cap2_2_2)

    # Skip connection
    up1_3 = layers.Concatenate(axis=-2, name='up1_3')([deconv_cap1_3_1, conv1_1_reshaped])
    up2_3 = layers.Concatenate(axis=-2, name='up2_3')([deconv_cap2_3_1, conv2_1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps1 = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps1')(up1_3)
    seg_caps2 = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                 routings=3, name='seg_caps2')(up2_3)



    _, H, W, C, A = seg_caps1.get_shape()
    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        out_recon = layers.Conv2D(filters=3, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_1)

        return out_recon



    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(finally_seg_caps)
    LengthOfseg_caps1 = Length(num_classes=n_class, seg=True, name='LengthOfseg_caps1')(seg_caps1)
    LengthOfseg_caps2 = Length(num_classes=n_class, seg=True, name='LengthOfseg_caps2')(seg_caps2)
    normMulpily = MultiplyLayer(num_classes=n_class, seg=True, name='normMulpily')([LengthOfseg_caps1,LengthOfseg_caps2])
    MultiplyVec = ReduceSumLayer(name="ReduceSumLayer")(MultiplyLayer(name='MultiplyLayerab')([seg_caps1, seg_caps2]))
    cos = DivideLayer(name='DivideLayer')([MultiplyVec,normMulpily])
    OutCos = OutCosLayer(num_classes=n_class, seg=True, name='OutCos')(cos)
    SubstractVec = layers.subtract([seg_caps1, seg_caps2])
    #
    # finally_SubstractVec = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
    #                             routings=3, name='finally_seg_caps')(SubstractVec)
    # Length_SubstractVec = Length(num_classes=n_class, seg=True, name='out_seg')(finally_SubstractVec)
    OutSubstrct = HalfLength(num_classes=n_class, seg=True, name='OutSubstrct')(SubstractVec)
    train_model=models.Model(inputs=[x1,x2], outputs=[OutSubstrct,OutCos,shared_decoder(seg_caps1)])
    # train_model=models.Model(inputs=[x1,x2], outputs=[OutCos,out_seg])
    return train_model
