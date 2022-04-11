from src.computer_vision.model_classification import ModelClassification
from src.computer_vision.model_cv import ModelCV
from src.computer_vision.model_data_performance import DataPerformanceConfiguration

DATASET_PATH = '/home/viti/Documents/UEB/Semestre9/SistemasInteligentes/ComputerVision/dataset/'


class ComputerVisionController:

    def __init__(self):
        self.model_classification = ModelClassification(DATASET_PATH)
        self.model_performance = DataPerformanceConfiguration()
        performance = self.model_performance.configure_data_performance(self.model_classification.train_ds,
                                                                        self.model_classification.val_ds)
        self.model_classification.train_ds = performance['train']
        self.model_classification.val_ds = performance['validate']

        # Normalize model
        self.model_classification.normalize_data()
        self.model = ModelCV(self.model_classification)

    def train(self):
        self.model.compile()
        self.model.summary()
        self.model.train_model()

    def predict(self, image_path: str, image_name: str):
        return self.model.predict(image_path, image_name)

    def save(self):
        self.model.save('shop_model.h5')

    def load(self):
        self.model.load('shop_model.h5')
