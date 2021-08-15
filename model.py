import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, DepthwiseConv2D, Conv2D, MaxPool2D, Add, Activation,ReLU
import time
import os

from custom_loss import smooth_l1_loss
#from dataloader import dataloader

def single_blaze_block(input_tensor,filters,strides=(1,1)):
    residual = input_tensor

    #Conv Layer
    x = DepthwiseConv2D((5,5), strides=strides, padding="same")(input_tensor)
    x = Conv2D(filters, (1,1), padding="same")(x)

    #Resiudal Stride & Padding
    pad_dim = filters - input_tensor.shape[3]
    pad_value = [[0,0], [0,0], [0,0], [0, pad_dim]]
    if strides[0] == 2:
        residual = MaxPool2D(pool_size=strides, strides=strides)(residual)
    if pad_dim != 0:
        residual = tf.pad(residual, pad_value)

    #Out Layer
    out = Add()([x, residual])
    act_out = ReLU()(out)
    return act_out


def double_blaze_block(input_tensor,filters,strides=(1,1)):
    residual = input_tensor

    #Conv Layer
    x = DepthwiseConv2D((5,5), strides=strides, padding="same")(input_tensor)
    x = Conv2D(filters[0], (1,1), padding="same")(x)
    x = ReLU()(x)
    x = DepthwiseConv2D((5,5), padding="same")(x)
    x = Conv2D(filters[1], (1,1), padding="same")(x)

    #Resiudal Stride & Padding
    pad_dim = filters[1] - input_tensor.shape[3]
    pad_value = [[0,0], [0,0], [0,0], [0, pad_dim]]
    if strides[0] == 2:
        residual = MaxPool2D(pool_size=strides, strides=strides)(residual)
    if pad_dim != 0:
        residual = tf.pad(residual, pad_value)

    #Out Layer
    out = Add()([x, residual])
    act_out = ReLU()(out)
    return act_out


def network(input_shape):
    #input layers
    inputs = Input(shape=input_shape)
    pre_conv_out = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding="SAME")(inputs)
    act_out = ReLU()(pre_conv_out)

    #single blaze blocks
    sbb_1 = single_blaze_block(act_out,24)
    sbb_2 = single_blaze_block(sbb_1,24)
    sbb_3 = single_blaze_block(sbb_2,48,strides=(2,2))
    sbb_4 = single_blaze_block(sbb_3,48)
    sbb_5 = single_blaze_block(sbb_4,48)

    #double blaze blocks
    dbb_1 = double_blaze_block(sbb_5,[24,96],strides=(2,2))
    dbb_2 = double_blaze_block(dbb_1,[24,96])
    dbb_3 = double_blaze_block(dbb_2,[24,96])
    dbb_4 = double_blaze_block(dbb_3,[24,96],strides=(2,2))
    dbb_5 = double_blaze_block(dbb_4,[24,96])
    dbb_6 = double_blaze_block(dbb_5,[24,96])

    model = tf.keras.models.Model(inputs=inputs, outputs=[dbb_3 , dbb_6])
    return model


class BlazeFace():
    def __init__(self,hyper_parameters):
        self.input_shape = hyper_parameters["input_shape"] 
        self.feature_extractor = network(self.input_shape)

        self.n_boxes = hyper_parameters["detections_per_layer"] # 2 for 16x16, 6 for 8x8
        self.model = self.build_model()

        if hyper_parameters["train"]:
            self.batch_size = hyper_parameters["batch_size"]
            self.nb_epoch = hyper_parameters["epoch"]
            
        self.checkpoint_path = hyper_parameters["checkpoint_path"]
        self.numdata = hyper_parameters["num_data"]

        self.dataset_dir = hyper_parameters["dataset_dir"]
        self.label_path= hyper_parameters["label_path"]

    def build_model(self):    
        model = self.feature_extractor
        
        # 16x16 bounding box - Confidence, [batch_size, 16, 16, 2]
        bb_16_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 1, 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='sigmoid')(model.output[0])
        # reshape [batch_size, 16**2 * #bbox(2), 1]
        bb_16_conf_reshaped = tf.keras.layers.Reshape((16**2 * self.n_boxes[0], 1))(bb_16_conf)
        
        
        # 8 x 8 bounding box - Confindece, [batch_size, 8, 8, 6]
        bb_8_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * 1, 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='sigmoid')(model.output[1])
        # reshape [batch_size, 8**2 * #bbox(6), 1]
        bb_8_conf_reshaped = tf.keras.layers.Reshape((8**2 * self.n_boxes[1], 1))(bb_8_conf)
        # Concatenate confidence prediction 
        
        # shape : [batch_size, 896, 1]
        conf_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_conf_reshaped, bb_8_conf_reshaped])
        
        
        # 16x16 bounding box - loc [x, y, w, h]
        bb_16_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 4,
                                            kernel_size=3, 
                                            padding='same')(model.output[0])
        # [batch_size, 16**2 * #bbox(2), 4]
        bb_16_loc_reshaped = tf.keras.layers.Reshape((16**2 * self.n_boxes[0], 4))(bb_16_loc)
        
        
        # 8x8 bounding box - loc [x, y, w, h]
        bb_8_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * 4,
                                        kernel_size=3,
                                        padding='same')(model.output[1])
        bb_8_loc_reshaped = tf.keras.layers.Reshape((8**2 * self.n_boxes[1], 4))(bb_8_loc)
        # Concatenate  location prediction 
        
        loc_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_loc_reshaped, bb_8_loc_reshaped])
        
        output_combined = tf.keras.layers.Concatenate(axis=-1)([conf_of_bb, loc_of_bb])
        
        # Detectors model 
        return tf.keras.models.Model(model.input, output_combined)


    def train(self,gen_train):
        opt = tf.keras.optimizers.Adam(amsgrad=True)
        model = self.model
        model.compile(loss=['categorical_crossentropy', smooth_l1_loss], optimizer=opt)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=4)

        ## Callback for Tensorboard ##
        tb = tf.keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')


        STEP_SIZE_TRAIN = self.numdata // self.batch_size

        t0 = time.time()

        #data_gen = dataloader(self.dataset_dir, self.label_path, self.batch_size)

        ## Train ##
        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=gen_train,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr, tb],
                                      verbose=1,
                                      shuffle=True)

        t2 = time.time()
            
        print(res.history)
        
        print('Training time for one epoch : %.1f' % ((t2 - t1)))

        if epoch % 100 == 0:
            model.save_weights(os.path.join(self.checkpoint_path,str(epoch)))

        print('Total training time : %.1f' % (time.time() - t0))