from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

model = load_model('cancer_classifier.h5')

