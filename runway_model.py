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

# =========================================================================

# This example contains the minimum specifications and requirements
# to port a machine learning model to Runway.

# For more instructions on how to port a model to Runway, see the Runway Model
# SDK docs at https://sdk.runwayml.com

# RUNWAY
# www.runwayml.com
# hello@runwayml.com

# =========================================================================

# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import number, text, image, vector
from example_model import CPPNModel

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://docs.runwayapp.ai/#/python-sdk to see a complete list of
# supported configs. The setup function should return the model ready to be
# used.

res = 64
setup_options = {
    'mode': text(default='tanh'),
    'seed': number(min=0, max=1000000, default=5, description='A seed used to initialize the model.'),
    'resolution': number(min=32, max=1024, default=64, description='output image size')
}
@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Ran with options: mode = {}, seed = {}, res = {}'
    print(msg.format(opts['mode'], opts['seed'], opts['resolution']))
    res = opts['resolution']
    model = CPPNModel(opts)
    return model


# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html
sample_inputs = {'z': vector(length=3)}
sample_outputs = {'image': image(width=res, height=res)}

@runway.command(name='generate', inputs=sample_inputs, outputs=sample_outputs,)
def generate(model, inputs):
    # print('[GENERATE] Ran with caption value "{}"'.format(args['caption']))
    # Generate a PIL or Numpy image based on the input caption, and return it
    output_image = model.run_on_input(inputs)
    return {
        'image': output_image
    }


if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8001)

## Now that the model is running, open a new terminal and give it a command to
## generate an image. It will respond with a base64 encoded URI
# curl \
#   -H "content-type: application/json" \
#   -d '{ "caption": "red" }' \
#   localhost:8000/generate
