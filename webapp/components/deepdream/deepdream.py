import cv2
from skimage.metrics import structural_similarity as ssim
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import inception_v3
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')


class DeepDreamGenerator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.layer_contributions = {
            'mixed3': 0.5,
            'mixed4': 3.0,
            'mixed5': 2.5,
            'mixed6': 0.5,
            'mixed7': 0.5,
        }
        self.base_model = inception_v3.InceptionV3(
            weights='imagenet', include_top=False)
        layer_outputs = [self.base_model.get_layer(
            name).output for name in self.layer_contributions.keys()]
        self.model = tf.keras.Model(
            inputs=self.base_model.input, outputs=layer_outputs)
        self.original_img = self.preprocess_image(image_path)
        self.original_img = tf.image.resize(self.original_img, (224, 224))

    def preprocess_image(self, image_path):
        img = image.load_img(image_path)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = inception_v3.preprocess_input(img)
        return img

    def save_image(self, img, filepath):
        img = np.array(img)
        img = np.squeeze(img)  # Remove the batch dimension
        img = 255 * (img + 1.0) / 2.0  # Convert from [-1, 1] to [0, 255]
        img = np.clip(img, 0, 255).astype('uint8')

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(filepath)
        print(f"Image saved at path {filepath}")

    @tf.function
    def deepdream_step(self, img, step_size):
        with tf.GradientTape() as tape:
            tape.watch(img)
            layer_activations = self.model(img)
            loss = tf.add_n([tf.reduce_mean(tf.square(act)) * self.layer_contributions[layer_name]
                             for act, layer_name in zip(layer_activations, self.layer_contributions.keys())])
        grads = tape.gradient(loss, img)
        grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
        img += grads * step_size
        return loss, img

    def run_deepdream_with_octaves(self):
        base_step_size = 0.01
        iterations = 20
        num_octaves = 3
        octave_scale = 1.4
        max_loss = 15.0

        img = self.original_img
        original_shape = tf.cast(tf.shape(img)[1:3], tf.float32)
        smallest_shape = tf.cast(
            original_shape * (octave_scale ** -num_octaves), tf.int32)
        img = tf.image.resize(img, smallest_shape)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        for n in range(num_octaves + 1):
            scale = octave_scale ** n
            new_size = tf.cast(original_shape * scale, tf.int32)
            img = tf.image.resize(img, new_size)
            for i in range(iterations):
                loss, img = self.deepdream_step(img, base_step_size)
                if loss > max_loss:
                    break
                print(f"Scale {scale}, Iteration: {i}, Loss: {loss}")
        return img

    def find_similarity_index(self, dream_img):
        # Rescale and convert the images for SSIM computation
        original_img = self.preprocess_image(self.image_path)
        original_rescaled = tf.image.resize(
            original_img, (256, 256)).numpy().squeeze()
        dream_rescaled = tf.image.resize(
            dream_img, (256, 256)).numpy().squeeze()
        original_rescaled = np.clip(
            original_rescaled * 255, 0, 255).astype('uint8')
        dream_rescaled = np.clip(dream_rescaled * 255, 0, 255).astype('uint8')
        gray_original = cv2.cvtColor(original_rescaled, cv2.COLOR_RGB2GRAY)
        gray_dream = cv2.cvtColor(dream_rescaled, cv2.COLOR_RGB2GRAY)

        # Compute SSIM
        score, _ = ssim(gray_original, gray_dream, data_range=255, full=True)
        print(
            f'Similarity between the original and  DeepDream image: {score:.2f}')
        return score

    def generate_dream(self):
        dream_img = self.run_deepdream_with_octaves()
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filepath = f'{self.image_path}_DeepDream_{timestamp}.png'
        filepath = filepath.replace("uploaded", "generated")
        self.save_image(dream_img, filepath)
        score = self.find_similarity_index(dream_img)
        score_percentage = round(score * 100, 2)
        return filepath, score_percentage
