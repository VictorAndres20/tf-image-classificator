from src.api.model.request import Request
from src.api.model.response import Response
from src.api.service.shop_service import ShopService


class ShopController:

    def __init__(self):
        self.service = ShopService()

    def get_product_info(self, req: Request):
        req = Response(
            code=201,
            ok=True,
            msg='Predicted',
            error='',
            data=self.service.get_product_info(req.image)
        )
        return req
