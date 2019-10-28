import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np
      
num_samples = 1000
height = 224
width = 224
num_classes = 1000
# Model weight may end up hosted on a GPU, which would complicate weight sharing.
with tf.device('/cpu:0'):
	model = Xception(weights=None,
	input_shape=(height, width, 3),
	classes=num_classes)

# Replicates the model on 8 GPUs.
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)

# Save model via the template model (which shares the same weights):
model.save('my_model.h5')
