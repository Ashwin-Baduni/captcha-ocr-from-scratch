"""
Advanced augmentation effects for CAPTCHA generation
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import cv2
from scipy import ndimage
from typing import Tuple, Optional
import random
import math

class CaptchaEffects:
    """Collection of effects to make CAPTCHAs more challenging"""

    @staticmethod
    def add_gaussian_noise(image: Image.Image, mean: float = 0, std: float = 0.01) -> Image.Image:
        """Add Gaussian noise to image"""
        img_array = np.array(image)
        noise = np.random.normal(mean, std * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    @staticmethod
    def add_salt_pepper_noise(image: Image.Image, amount: float = 0.01) -> Image.Image:
        """Add salt and pepper noise"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        num_pixels = int(amount * h * w)

        coords = [np.random.randint(0, i - 1, num_pixels) for i in img_array.shape[:2]]
        img_array[coords[0], coords[1]] = 255

        coords = [np.random.randint(0, i - 1, num_pixels) for i in img_array.shape[:2]]
        img_array[coords[0], coords[1]] = 0

        return Image.fromarray(img_array)

    @staticmethod
    def add_speckle_noise(image: Image.Image, variance: float = 0.01) -> Image.Image:
        """Add speckle noise (multiplicative noise)"""
        img_array = np.array(image).astype(np.float32) / 255
        noise = np.random.randn(*img_array.shape) * variance
        noisy_img = img_array + img_array * noise
        noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    @staticmethod
    def add_motion_blur(image: Image.Image, size: int = 5, angle: float = 0) -> Image.Image:
        """Add motion blur effect"""
        img_array = np.array(image)

        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size

        rotation_matrix = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))

        if len(img_array.shape) == 3:
            blurred = np.stack([
                cv2.filter2D(img_array[:,:,i], -1, kernel)
                for i in range(img_array.shape[2])
            ], axis=2)
        else:
            blurred = cv2.filter2D(img_array, -1, kernel)

        return Image.fromarray(blurred)

    @staticmethod
    def elastic_distortion(image: Image.Image, alpha: float = 30, sigma: float = 5) -> Image.Image:
        """Apply elastic distortion to image"""
        img_array = np.array(image)
        shape = img_array.shape[:2]

        dx = np.random.randn(*shape) * sigma
        dy = np.random.randn(*shape) * sigma

        dx = cv2.GaussianBlur(dx, (5, 5), sigma)
        dy = cv2.GaussianBlur(dy, (5, 5), sigma)

        dx = dx * alpha
        dy = dy * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x_distorted = x + dx
        y_distorted = y + dy

        if len(img_array.shape) == 3:
            distorted = np.zeros_like(img_array)
            for i in range(img_array.shape[2]):
                distorted[:,:,i] = cv2.remap(
                    img_array[:,:,i],
                    x_distorted.astype(np.float32),
                    y_distorted.astype(np.float32),
                    cv2.INTER_LINEAR
                )
        else:
            distorted = cv2.remap(
                img_array,
                x_distorted.astype(np.float32),
                y_distorted.astype(np.float32),
                cv2.INTER_LINEAR
            )

        return Image.fromarray(distorted)

    @staticmethod
    def add_sine_wave_distortion(image: Image.Image, amplitude: float = 5, frequency: float = 0.05) -> Image.Image:
        """Add sine wave distortion"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        displacement = amplitude * np.sin(2 * np.pi * frequency * x)

        y_distorted = y + displacement

        if len(img_array.shape) == 3:
            distorted = np.zeros_like(img_array)
            for i in range(img_array.shape[2]):
                distorted[:,:,i] = cv2.remap(
                    img_array[:,:,i],
                    x.astype(np.float32),
                    y_distorted.astype(np.float32),
                    cv2.INTER_LINEAR
                )
        else:
            distorted = cv2.remap(
                img_array,
                x.astype(np.float32),
                y_distorted.astype(np.float32),
                cv2.INTER_LINEAR
            )

        return Image.fromarray(distorted)

    @staticmethod
    def add_security_lines(image: Image.Image, num_lines: int = 3, thickness: int = 2) -> Image.Image:
        """Add random lines through the text (like real CAPTCHAs)"""
        draw = ImageDraw.Draw(image)
        w, h = image.size

        for _ in range(num_lines):

            x1 = random.randint(0, w // 4)
            y1 = random.randint(h // 4, 3 * h // 4)
            x2 = random.randint(3 * w // 4, w)
            y2 = random.randint(h // 4, 3 * h // 4)

            color = random.randint(100, 200)

            points = []
            for i in range(10):
                t = i / 9
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1) + random.randint(-5, 5)
                points.append((x, y))

            for i in range(len(points) - 1):
                draw.line([points[i], points[i+1]], fill=(color, color, color), width=thickness)

        return image

    @staticmethod
    def add_dots_pattern(image: Image.Image, density: float = 0.02) -> Image.Image:
        """Add random dots pattern as background noise"""
        draw = ImageDraw.Draw(image)
        w, h = image.size
        num_dots = int(w * h * density)

        for _ in range(num_dots):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            radius = random.randint(1, 3)
            color = random.randint(150, 230)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                        fill=(color, color, color))

        return image

    @staticmethod
    def add_grid_pattern(image: Image.Image, spacing: int = 20, alpha: float = 0.3) -> Image.Image:
        """Add grid pattern overlay"""
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = image.size

        for x in range(0, w, spacing):
            color = (200, 200, 200, int(255 * alpha))
            draw.line([(x, 0), (x, h)], fill=color, width=1)

        for y in range(0, h, spacing):
            color = (200, 200, 200, int(255 * alpha))
            draw.line([(0, y), (w, y)], fill=color, width=1)

        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return Image.alpha_composite(image, overlay).convert('RGB')

    @staticmethod
    def perspective_transform(image: Image.Image, intensity: float = 0.0002) -> Image.Image:
        """Apply perspective transformation"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        offset = int(intensity * w * h)
        dst_points = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)]
        ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        transformed = cv2.warpPerspective(img_array, matrix, (w, h))

        return Image.fromarray(transformed)

    @staticmethod
    def random_rotation(image: Image.Image, max_angle: float = 5) -> Image.Image:
        """Rotate image by random angle"""
        angle = random.uniform(-max_angle, max_angle)
        return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    @staticmethod
    def random_shear(image: Image.Image, max_shear: float = 0.2) -> Image.Image:
        """Apply random shear transformation"""
        shear = random.uniform(-max_shear, max_shear)

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        shear_matrix = np.array([
            [1, shear, -shear * h / 2],
            [0, 1, 0]
        ], dtype=np.float32)

        sheared = cv2.warpAffine(img_array, shear_matrix, (w, h),
                                 borderValue=(255, 255, 255))

        return Image.fromarray(sheared)

    @staticmethod
    def add_texture_background(image: Image.Image, texture_type: str = 'paper') -> Image.Image:
        """Add textured background"""
        w, h = image.size

        if texture_type == 'paper':

            texture = Image.new('RGB', (w, h), (245, 245, 240))
            noise = np.random.randint(235, 255, (h, w, 3), dtype=np.uint8)
            texture = Image.fromarray(noise)
        elif texture_type == 'canvas':

            texture = Image.new('RGB', (w, h), (250, 248, 245))
            draw = ImageDraw.Draw(texture)
            for _ in range(100):
                x, y = random.randint(0, w), random.randint(0, h)
                color = random.randint(240, 255)
                draw.point((x, y), fill=(color, color-2, color-5))
        else:
            texture = Image.new('RGB', (w, h), (255, 255, 255))

        return Image.blend(texture, image, 0.7)

    @staticmethod
    def character_level_distortion(image: Image.Image, text: str) -> Image.Image:
        """Apply different distortions to individual characters (advanced)"""

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        num_chars = len(text)
        if num_chars > 0:
            region_width = w // num_chars

            for i in range(num_chars):
                x_start = i * region_width
                x_end = min((i + 1) * region_width, w)

                if random.random() > 0.5:

                    angle = random.uniform(-3, 3)
                    region = img_array[:, x_start:x_end]
                    region = ndimage.rotate(region, angle, reshape=False, cval=255)
                    img_array[:, x_start:x_end] = region

        return Image.fromarray(img_array)