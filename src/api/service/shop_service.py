import os.path

from src.api.utils.image_handler import write_image
from src.computer_vision.controller import ComputerVisionController


class ShopService:

    def __init__(self):
        self.info = None
        self.build_info()
        self.cv = ComputerVisionController()
        self.cv.load()

    def get_product_info(self, image: str):
        image_name = "predict.png"
        write_image(image, os.path.join(os.path.dirname(os.path.abspath(__file__)), image_name))
        result_dict = self.cv.predict(os.path.dirname(os.path.abspath(__file__)), image_name)
        return self.info[result_dict["class"]]

    def build_info(self):
        self.info = {
            "banana": {
                "description": "Banano de importación nacional",
                "stock": 20,
                "price": 1300
            },
            "choclitos": {
                "description": "Choclitos x 50 gramos",
                "stock": 45,
                "price": 1000
            },
            "poker": {
                "description": "Poker lata 500ml",
                "stock": 33,
                "price": 3500
            },
            "reloj": {
                "description": "Reloj Swatch de importación internacional",
                "stock": 5,
                "price": 450000
            }
        }

