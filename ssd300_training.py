#!/usr/bin/env python
# coding: utf-8

# # SSD300 Training Tutorial
# 
# This tutorial explains how to train an SSD300 on the Pascal VOC datasets. The preset parameters reproduce the training of the original SSD300 "07+12" model. Training SSD512 works simiarly, so there's no extra tutorial for that. The same goes for training on other datasets.
# 
# You can find a summary of a full training here to get an impression of what it should look like:
# [SSD300 "07+12" training summary](https://github.com/pierluigiferrari/ssd_keras/blob/master/training_summaries/ssd300_pascal_07%2B12_training_summary.md)

# In[ ]:


from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
#from keras.utils.training_utils import multi_gpu_model
from keras.utils.training_utils import multi_gpu_model

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

import argparse
# ## 0. Preliminary note
# 
# All places in the code where you need to make any changes are marked `TODO` and explained accordingly. All code cells that don't contain `TODO` markers just need to be executed.

# ## 1. Set the model configuration parameters
# 
# This section sets the configuration parameters for the model definition. The parameters set here are being used both by the `ssd_300()` function that builds the SSD300 model as well as further down by the constructor for the `SSDInputEncoder` object that is needed to run the training. Most of these parameters are needed to define the anchor boxes.
# 
# The parameters as set below produce the original SSD300 architecture that was trained on the Pascal VOC datsets, i.e. they are all chosen to correspond exactly to their respective counterparts in the `.prototxt` file that defines the original Caffe implementation. Note that the anchor box scaling factors of the original SSD implementation vary depending on the datasets on which the models were trained. The scaling factors used for the MS COCO datasets are smaller than the scaling factors used for the Pascal VOC datasets. The reason why the list of scaling factors has 7 elements while there are only 6 predictor layers is that the last scaling factor is used for the second aspect-ratio-1 box of the last predictor layer. Refer to the documentation for details.
# 
# As mentioned above, the parameters set below are not only needed to build the model, but are also passed to the `SSDInputEncoder` constructor further down, which is responsible for matching and encoding ground truth boxes and anchor boxes during the training. In order to do that, it needs to know the anchor box parameters.

# In[2]:

parser = argparse.ArgumentParser()
parser.add_argument('--gpus',help='number of gpus')
args = parser.parse_args()

if args.gpus:
    gpu_count = int(args.gpus)
img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
# n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
num_train = 1000

# 実験用クラス
classes = ['background',
        'bus', 'car', 'cat',
        'horse', 'motorbike']
n_classes = len(classes) - 1

# ## 2. Build or load the model
# 
# You will want to execute either of the two code cells in the subsequent two sub-sections, not both.

# ### 2.1 Create a new model and load trained VGG-16 weights into it (or trained SSD weights)
# 
# If you want to create a new SSD300 model, this is the relevant section for you. If you want to load a previously saved SSD300 model, skip ahead to section 2.2.
# 
# The code cell below does the following things:
# 1. It calls the function `ssd_300()` to build the model.
# 2. It then loads the weights file that is found at `weights_path` into the model. You could load the trained VGG-16 weights or you could load the weights of a trained model. If you want to reproduce the original SSD training, load the pre-trained VGG-16 weights. In any case, you need to set the path to the weights file you want to load on your local machine. Download links to all the trained weights are provided in the [README](https://github.com/pierluigiferrari/ssd_keras/blob/master/README.md) of this repository.
# 3. Finally, it compiles the model for the training. In order to do so, we're defining an optimizer (Adam) and a loss function (SSDLoss) to be passed to the `compile()` method.
# 
# Normally, the optimizer of choice would be Adam (commented out below), but since the original implementation uses plain SGD with momentum, we'll do the same in order to reproduce the original training. Adam is generally the superior optimizer, so if your goal is not to have everything exactly as in the original training, feel free to switch to Adam. You might need to adjust the learning rate scheduler below slightly in case you use Adam.
# 
# Note that the learning rate that is being set here doesn't matter, because further below we'll pass a learning rate scheduler to the training function, which will overwrite any learning rate set here, i.e. what matters are the learning rates that are defined by the learning rate scheduler.
# 
# `SSDLoss` is a custom Keras loss function that implements the multi-task that consists of a log loss for classification and a smooth L1 loss for localization. `neg_pos_ratio` and `alpha` are set as in the paper.

# In[4]:


# 1: Build the Keras model.

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)



# TODO: Set the path to the weights you want to load.
weights_path = 'VGG_ILSVRC_16_layers_fc_reduced.h5'
model.load_weights(weights_path, by_name=True)
# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)
model.summary()
# In[5]:


# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
"""
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='dataset_pascal_voc_07+12_half_trainval.h5')
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='dataset_pascal_voc_07_half_test.h5')
"""
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)


# if train_dataset.hdf5_dataset_path and val_dataset.hdf5_dataset_path:
#     train_dataset.load_hdf5_dataset()
#     val_dataset.load_hdf5_dataset()

if False:
    print('a')
else:
    # 2: Parse the image and label lists for the training and validation datasets. This can take a while.

    # TODO: Set the paths to the datasets here.

    # The directories that contain the images.
    VOC_2007_images_dir      = '/home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages/'
    VOC_2012_images_dir      = '/home/ubuntu/data/VOCdevkit/VOC2012/JPEGImages/'

    # The directories that contain the annotations.
    VOC_2007_annotations_dir      = '/home/ubuntu/data/VOCdevkit/VOC2007/Annotations/'
    VOC_2012_annotations_dir      = '/home/ubuntu/data/VOCdevkit/VOC2012/Annotations/'

    # The paths to the image sets.
    # VOC_2007_train_image_set_filename    = '/home/ubuntu/data/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
    # VOC_2012_train_image_set_filename    = '/home/ubuntu/data/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
    # VOC_2007_val_image_set_filename      = '/home/ubuntu/data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
    # VOC_2012_val_image_set_filename      = '/home/ubuntu/data/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
    # VOC_2007_trainval_image_set_filename = '/home/ubuntu/data/VOCdevkit/VOC2007/ImageSets/Main/2007_class_limitted.txt'
    # VOC_2012_trainval_image_set_filename = '/home/ubuntu/data/VOCdevkit/VOC2012/ImageSets/Main/2012_class_limitted.txt'
    # VOC_2007_test_image_set_filename     = '/home/ubuntu/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

    train_classlimitted_image_set_filename = '/home/ubuntu/data/VOCdevkit/VOC2012/ImageSets/Main/{}_class_limitted.txt'.format(num_train)
    # VOC_2012_train_classlimitted_image_set_filename = '/home/ubuntu/data/VOCdevkit/VOC2012/ImageSets/Main/2012_train_class_limitted.txt'
    valid_image_set_filename     = '/home/ubuntu/data/VOCdevkit/VOC2007/ImageSets/Main/{}_class_limitted.txt'.format(num_train)
    # VOC_2012_valid_image_set_filename     = '/home/ubuntu/data/VOCdevkit/VOC2007/ImageSets/Main/2012_valid_class_limitted.txt'

    # The XML parser needs to now what object class names to look for and in which order to map them to integers.
    """
    classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
    """

    train_dataset.parse_xml(images_dirs=[VOC_2012_images_dir],
                            image_set_filenames=[train_classlimitted_image_set_filename],
                                                #VOC_2012_train_classlimitted_image_set_filename],
                            annotations_dirs=[VOC_2012_annotations_dir],
                                            #VOC_2012_annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                        image_set_filenames=[valid_image_set_filename],
                        annotations_dirs=[VOC_2007_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=True,
                        ret=False,
                        valid=True)

    # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
    # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
    # option in the constructor, because in that cas the images are in memory already anyway. If you don't
    # want to create HDF5 datasets, comment out the subsequent two function calls.

    # train_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07+12_classlimitted_trainval.h5',
    #                                 resize=False,
    #                                 variable_image_size=True,
    #                                 verbose=True)

    # val_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07_classlimitted_test.h5',
    #                                 resize=False,
    #                                 variable_image_size=True,
    #                                 verbose=True)

# In[6]:


# 3: Set the batch size.

batch_size =32  # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer(name='conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer(name='fc7_mbox_conf').output_shape[1:3],
                   model.get_layer(name='conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer(name='conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer(name='conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer(name='conv9_2_mbox_conf').output_shape[1:3]]
"""
parallel_model = multi_gpu_model(model, gpus=gpu_count) #複数gpu対応
parallel_model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)
parallel_model.summary()
"""
ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


# ## 4. Set the remaining training parameters
# 
# We've already chosen an optimizer and set the batch size above, now let's set the remaining training parameters. I'll set one epoch to consist of 1,000 training steps. The next code cell defines a learning rate schedule that replicates the learning rate schedule of the original Caffe implementation for the training of the SSD300 Pascal VOC "07+12" model. That model was trained for 120,000 steps with a learning rate of 0.001 for the first 80,000 steps, 0.0001 for the next 20,000 steps, and 0.00001 for the last 20,000 steps. If you're training on a different dataset, define the learning rate schedule however you see fit.
# 
# I'll set only a few essential Keras callbacks below, feel free to add more callbacks if you want TensorBoard summaries or whatever. We obviously need the learning rate scheduler and we want to save the best models during the training. It also makes sense to continuously stream our training history to a CSV log file after every epoch, because if we didn't do that, in case the training terminates with an exception at some point or if the kernel of this Jupyter notebook dies for some reason or anything like that happens, we would lose the entire history for the trained epochs. Finally, we'll also add a callback that makes sure that the training terminates if the loss becomes `NaN`. Depending on the optimizer you use, it can happen that the loss becomes `NaN` during the first iterations of the training. In later iterations it's less of a risk. For example, I've never seen a `NaN` loss when I trained SSD using an Adam optimizer, but I've seen a `NaN` loss a couple of times during the very first couple of hundred training steps of training a new model when I used an SGD optimizer.

# In[ ]:


# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 80:
        return 0.0001 #0.001 originally
    elif epoch < 100:
        return 0.00001
    else:
        return 0.000001


# In[ ]:


# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.

model_checkpoint = ModelCheckpoint(filepath='checkpoints_classlimitted/ssd300_pascal_07+12_1000_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd300_pascal_07+12_checkpoints_classlimitted_{}_training_log.csv'.format(num_train),
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

tensorboard = TensorBoard(log_dir='tflog/')

callbacks = [model_checkpoint, #multi_gpu_modelだとepoch毎の保存エラーになるので学習後に一度だけ保存する
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan,
             tensorboard]


# ## 5. Train

# In order to reproduce the training of the "07+12" model mentioned above, at 1,000 training steps per epoch you'd have to train for 120 epochs. That is going to take really long though, so you might not want to do all 120 epochs in one go and instead train only for a few epochs at a time. You can find a summary of a full training [here](https://github.com/pierluigiferrari/ssd_keras/blob/master/training_summaries/ssd300_pascal_07%2B12_training_summary.md).
# 
# In order to only run a partial training and resume smoothly later on, there are a few things you should note:
# 1. Always load the full model if you can, rather than building a new model and loading previously saved weights into it. Optimizers like SGD or Adam keep running averages of past gradient moments internally. If you always save and load full models when resuming a training, then the state of the optimizer is maintained and the training picks up exactly where it left off. If you build a new model and load weights into it, the optimizer is being initialized from scratch, which, especially in the case of Adam, leads to small but unnecessary setbacks every time you resume the training with previously saved weights.
# 2. In order for the learning rate scheduler callback above to work properly, `fit_generator()` needs to know which epoch we're in, otherwise it will start with epoch 0 every time you resume the training. Set `initial_epoch` to be the next epoch of your training. Note that this parameter is zero-based, i.e. the first epoch is epoch 0. If you had trained for 10 epochs previously and now you'd want to resume the training from there, you'd set `initial_epoch = 10` (since epoch 10 is the eleventh epoch). Furthermore, set `final_epoch` to the last epoch you want to run. To stick with the previous example, if you had trained for 10 epochs previously and now you'd want to train for another 10 epochs, you'd set `initial_epoch = 10` and `final_epoch = 20`.
# 3. In order for the model checkpoint callback above to work correctly after a kernel restart, set `model_checkpoint.best` to the best validation loss from the previous training. If you don't do this and a new `ModelCheckpoint` object is created after a kernel restart, that object obviously won't know what the last best validation loss was, so it will always save the weights of the first epoch of your new training and record that loss as its new best loss. This isn't super-important, I just wanted to mention it.

# In[ ]:


# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 150
steps_per_epoch = 100

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)

model.save("checkpoints_classlimitted/fitted_classlimitted_{}_model.h5".format(num_train))
