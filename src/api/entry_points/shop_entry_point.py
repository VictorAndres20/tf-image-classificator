from fastapi import APIRouter
from src.api.model.response import Response
from src.api.model.request import Request
from src.api.controller import shop_controller

controller = shop_controller.ShopController()

router = APIRouter(
    prefix="/shop",
    responses={
        404: {"Description": "Not found"}
    }
)


@router.post("/info")
async def find(req: Request) -> Response:
    return controller.get_product_info(req)
