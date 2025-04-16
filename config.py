import os

# Data paths
DATA_CSV = 'merged_CXLSeg_data.csv'
MODEL_SAVE_PATH = 'models'
LOGS_PATH = 'logs'

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Create necessary directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Labels for classification
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
] 