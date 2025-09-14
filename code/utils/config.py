"""
Configuration settings for CAPTCHA dataset generation
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
EASY_DIR = DATA_DIR / "easy"
HARD_DIR = DATA_DIR / "hard"
BONUS_DIR = DATA_DIR / "bonus"
METADATA_DIR = DATA_DIR / "metadata"

DATASET_SIZE = {
    'easy': 1000,
    'hard': 1000,
    'bonus': 1000
}

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 60
DPI = 100

FONTS = {
    'easy': ['DejaVu Sans'],
    'hard': [
        'DejaVu Sans',
        'DejaVu Serif',
        'DejaVu Sans Mono',
        'Liberation Sans',
        'Liberation Serif',
        'Liberation Mono'
    ]
}

FONT_SIZE_RANGE = {
    'easy': (28, 32),
    'hard': (20, 36),
    'bonus': (20, 36)
}

COLORS = {
    'easy': {
        'bg': (255, 255, 255),
        'text': (0, 0, 0)
    },
    'hard': {
        'bg_range': [(200, 255), (200, 255), (200, 255)],
        'text_range': [(0, 100), (0, 100), (0, 100)]
    },
    'bonus': {
        'green_bg': (144, 238, 144),
        'red_bg': (255, 182, 193)
    }
}

NOISE_LEVELS = {
    'easy': 0.0,
    'hard': {
        'gaussian': (0, 0.02),
        'salt_pepper': (0, 0.01),
        'speckle': (0, 0.015)
    },
    'bonus': 'same_as_hard'
}

DISTORTION_PARAMS = {
    'rotation': (-5, 5),
    'shear': (-0.2, 0.2),
    'perspective': 0.0002,
    'elastic': {
        'alpha': 30,
        'sigma': 5
    }
}

MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 10

WORD_LIST = [

    'cat', 'dog', 'run', 'top', 'hat', 'box', 'pen', 'red', 'big', 'hot',

    'book', 'tree', 'sand', 'home', 'work', 'play', 'food', 'rain', 'moon', 'star',
    'fish', 'bird', 'hand', 'foot', 'head', 'door', 'time', 'year', 'life', 'love',

    'water', 'house', 'happy', 'green', 'light', 'night', 'smile', 'dream', 'music', 'dance',
    'heart', 'world', 'peace', 'magic', 'power', 'story', 'cloud', 'ocean', 'forest', 'flower',

    'orange', 'purple', 'silver', 'friend', 'family', 'school', 'garden', 'planet', 'summer', 'winter',
    'spring', 'autumn', 'castle', 'bridge', 'window', 'nature', 'beauty', 'wisdom', 'courage', 'future',

    'freedom', 'journey', 'mystery', 'rainbow', 'sunrise', 'moonlit', 'diamond', 'crystal', 'thunder', 'magical',

    'mountain', 'computer', 'keyboard', 'sunshine', 'starlight', 'universe', 'elephant', 'butterfly', 'festival', 'treasure',

    'adventure', 'beautiful', 'wonderful', 'chocolate', 'pineapple', 'happiness', 'celebrate', 'discovery', 'knowledge', 'telephone',

    'strawberry', 'watermelon', 'basketball', 'technology', 'impossible', 'incredible', 'generation', 'revolution', 'atmosphere', 'helicopter'
]

DIFFICULTY_WEIGHTS = {
    'noise_level': 0.2,
    'contrast': 0.15,
    'distortion': 0.25,
    'font_complexity': 0.1,
    'background_complexity': 0.15,
    'text_overlap': 0.15
}

RANDOM_SEED = 42

SAVE_METADATA = True
METADATA_FORMAT = 'json'

VALIDATE_GENERATION = True
MIN_CONTRAST_RATIO = 2.0