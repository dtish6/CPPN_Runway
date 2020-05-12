# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from PIL import Image
import tensorflow as tf
import numpy as np


class CPPNModel:

    def __init__(self, options):
        self.seed = options['seed']
        self.mode = options['mode']
        self.res = options['resolution']

    def init_model_tanh(self):
        model = tf.keras.Sequential()
        init = tf.keras.initializers.VarianceScaling(scale=10, mode='fan_in', seed=self.seed)
        model.add(tf.keras.layers.Dense(32, activation="tanh", input_shape=(5,), kernel_initializer=init, use_bias=False))
        model.add(tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=init, use_bias=False))
        model.add(tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=init, use_bias=False))
        # model.add(layers.Dense(32, activation="tanh", kernel_initializer=init,use_bias=False))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=init, use_bias=False))
        return model

    def init_model_softplus(self):
        model = tf.keras.Sequential()
        init = tf.keras.initializers.VarianceScaling(scale=10, mode='fan_in', seed=self.seed)
        model.add(tf.keras.layers.Dense(32, activation="tanh", input_shape=(5,), kernel_initializer=init, use_bias=False))
        model.add(tf.keras.layers.Dense(32, activation="softplus", kernel_initializer=init, use_bias=False))
        model.add(tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=init, use_bias=False))
        model.add(tf.keras.layers.Dense(32, activation="softplus", kernel_initializer=init, use_bias=False))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=init, use_bias=False))
        return model

    def pixelGrid(self, resolution=64):
        x = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, x)
        return np.vstack([X.flatten(), Y.flatten()]).T

    def generateIm(self, model, z=[0, 0], resolution=64, scale=1.0):
        pixels = self.pixelGrid(resolution) * scale
        input = np.hstack(
            [pixels, np.linalg.norm(pixels, axis=1).reshape(-1, 1), np.repeat([z], resolution ** 2, axis=0) * scale])
        return model.predict(input, batch_size=128).reshape(resolution, resolution)


    # Generate an image based on input vector.
    def run_on_input(self, input_vec):

        # This is an example of how you could use some input from
        # @runway.setup(), like options['truncation'], later inside a
        # function called by @runway.command().
        # text = caption_text[0:self.truncation]

        if self.mode == 'tanh':
            model = self.init_model_tanh()
        elif self.mode == 'softplus':
            model = self.init_model_softplus()
        else:
            print("Provide either 'tanh' or 'softplus' as the mode")
            return None

        z1,z2,scale = input_vec
        RGB = np.zeros((self.res, self.res, 3), dtype=float)

        im1 = self.generateIm(model, z=[z1, z2], scale=scale, resolution=self.res)
        RGB[..., 0] = im1
        RGB[..., 1] = im1
        RGB[..., 2] = im1

        return Image.fromarray(np.uint8(RGB*255))

        # Return a red image if the input text is "red",
        # otherwise return a blue image.
        # if text == 'red':
        #     return Image.new('RGB', (512, 512), color = 'red')
        # else:
        #     return Image.new('RGB', (512, 512), color = 'blue')
