import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mtcnn import MTCNN
import datetime  # For generating unique file names
# Import the SSIM function
from skimage.metrics import structural_similarity as compare_ssim


class GanImageGenerator:
    def __init__(self, generator_model_path, image_path):
        print("Inside GENERATOR", generator_model_path)
        self.generator = tf.keras.models.load_model(generator_model_path)
        print("Model Loaded")
        self.image_path = image_path
        self.size = 96
        self.expansion_factor = 0.2
        self.detector = MTCNN()

    def preprocess_and_detect(self, image_path):
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (self.size, self.size))
        detections = self.detector.detect_faces(resized_image)

        if len(detections) > 0:
            x, y, w, h = detections[0]['box']
            exp_w, exp_h = int(
                w * self.expansion_factor), int(h * self.expansion_factor)
            x1, y1 = max(x - exp_w, 0), max(y - exp_h, 0)
            x2, y2 = min(x + w + exp_w, self.size), min(y +
                                                        h + exp_h, self.size)
            face = resized_image[y1:y2, x1:x2]
            return face / 255.0  # Normalize to [0, 1]
        return None

    def generate_cartoon(self, original_image, filepath):
        if original_image is None:
            print("No valid image detected or provided.")
            return None
        if len(original_image.shape) == 3:
            original_image = np.expand_dims(original_image, axis=0)

        noise_vector = np.random.normal(size=(1, 100))
        cartoon_image = self.generator.predict(noise_vector)

        cartoon_image_normalized = np.squeeze(cartoon_image[0])
        if cartoon_image_normalized.min() < 0 or cartoon_image_normalized.max() > 1:
            cartoon_image_normalized = (cartoon_image_normalized - cartoon_image_normalized.min()) / (
                cartoon_image_normalized.max() - cartoon_image_normalized.min())

        # Resize original and cartoon images for similarity comparison
        original_resized = cv2.resize(original_image[0], (256, 256))
        cartoon_resized = cv2.resize(cartoon_image_normalized, (256, 256))

        # Save the resized cartoon for consistency
        plt.figure(figsize=(8, 8))
        plt.imshow(cartoon_resized)
        plt.axis('off')
        plt.savefig(filepath)
        print(f'Cartoon image saved as {filepath}')

        # Convert to grayscale for SSIM calculation
        gray_original = cv2.cvtColor(
            (original_resized * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
        gray_cartoon = cv2.cvtColor(
            (cartoon_resized * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)

        # Calculate SSIM
        score, _ = compare_ssim(gray_original, gray_cartoon, full=True)
        score_percentage = round(score * 100, 2)
        return filepath, score_percentage

    def generate_gan_image(self):
        preprocessed_input_image = self.preprocess_and_detect(self.image_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filepath = f'{self.image_path}_gan_{timestamp}.png'
        filepath = filepath.replace("uploaded", "generated")
        # The score is returned by generate_cartoon
        return self.generate_cartoon(preprocessed_input_image, filepath)
