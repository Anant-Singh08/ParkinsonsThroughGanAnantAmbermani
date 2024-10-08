from flask import Flask, render_template, request, jsonify, json, send_file, send_from_directory, url_for
from time import sleep
import threading
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, LeakyReLU, ReLU , Dropout, UpSampling2D, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
import zipfile
import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
output_folder = r'user_output\brainscans'
# original_folder = '.\\JPG_Images'
output_folder_spiral_unhealthy = r'user_output\unhealthy_spiral'
output_folder_healthy_spiral = r'user_output\healthy_spiral'
# Made by Anant Singh and Ambermani Jha
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 1024)))
    assert model.output_shape == (None, 7, 7, 1024)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 112, 112,32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 224, 224, 1)

    return model
generator = make_generator_model()
gen_spiral = make_generator_model()
gen_spiral_healthy = make_generator_model()
generator.summary()
generator.load_weights(r'Models\generator.h5')
gen_spiral.load_weights(r'Models\generator_unhealthy.h5')
gen_spiral_healthy.load_weights(r'Models\generator_healthy.h5')

noise_dim = 100
num_examples_to_generate = 16


seed = tf.random.normal([num_examples_to_generate, noise_dim])
def generate_and_save_images(model, number, test_input):
    for num in range(number):
        prediction = model(test_input, training=False)

        # Extract the image (assuming grayscale image)
        generated_image = prediction[0, :, :, 0] * 127.5 + 127.5

        # Create a figure
        fig = plt.figure(figsize=(12.8, 12.8), dpi=100)  # 128 by 128 pixels at 100 DPI

        #Displaying
        plt.imshow(generated_image, cmap='gray')
        plt.axis('off')

        # Saving
        plt.savefig(os.path.join(output_folder, f'image_at_epoch_{num:04d}.png'), bbox_inches='tight', pad_inches=0)
    return render_template("index.html", status = f'image_at_epoch_{num:04d}.png')

def generate_and_save_images_unhealthy_spiral(model, number, test_input):
    for num in range(number):
        prediction = model(test_input, training=False)

        # Extract the image (assuming grayscale image)
        generated_image = prediction[0, :, :, 0] * 127.5 + 127.5

        # Create a figure
        fig = plt.figure(figsize=(12.8, 12.8), dpi=100)  # 128 by 128 pixels at 100 DPI

        #Displaying
        plt.imshow(generated_image, cmap='gray')
        plt.axis('off')

        # Saving
        plt.savefig(os.path.join(output_folder_spiral_unhealthy, f'image_at_epoch_{num:04d}.png'), bbox_inches='tight', pad_inches=0)
def generate_and_save_images_healthy_spiral(model, number, test_input):
    for num in range(number):
        prediction = model(test_input, training=False)

        # Extract the image (assuming grayscale image)
        generated_image = prediction[0, :, :, 0] * 127.5 + 127.5

        # Create a figure
        fig = plt.figure(figsize=(12.8, 12.8), dpi=100)  # 128 by 128 pixels at 100 DPI

        #Displaying
        plt.imshow(generated_image, cmap='gray')
        plt.axis('off')

        # Saving
        plt.savefig(os.path.join(output_folder_healthy_spiral, f'image_at_epoch_{num:04d}.png'), bbox_inches='tight', pad_inches=0)
number = 50
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def run_task(generator, number, seed, gen_spiral=None, gen_spiral_healthy = None):
    try:
        generate_and_save_images(generator, number, seed)
        if gen_spiral:
            generate_and_save_images_unhealthy_spiral(gen_spiral, number, seed)
        if gen_spiral_healthy:
            generate_and_save_images_healthy_spiral(gen_spiral_healthy, number, seed)
        return app.json.response({'message': 'ZIP file generation complete!'})  
    except Exception as e:
        return app.json.response({'error': str(e)})

@app.route('/start_task', methods=['POST', 'GET'])
def start_task():
    run_task(generator, number, seed, gen_spiral, gen_spiral_healthy)
    return jsonify({'message': 'Task started in a separate thread.'})

UPLOAD_FOLDER = ''

@app.route('/download', methods=['GET'])
def download():
    """Handles file download requests."""
    folder_to_zip = "user_output"
    zip_file = "output.zip"
    with zipfile.ZipFile(zip_file, 'w') as zip_obj:
        for root, dirs, files in os.walk(folder_to_zip):
         for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, folder_to_zip)
            zip_obj.write(file_path,  arcname)
    try:
        file_path = os.path.join(app.root_path, UPLOAD_FOLDER, 'output.zip')
        if not os.path.isfile(file_path):
            return "File not found", 404
        return send_from_directory(UPLOAD_FOLDER, 'output.zip')

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
