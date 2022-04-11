from src.api.entry_points import shop_entry_point
from fastapi import FastAPI


def set_api_routes(app: FastAPI):
    app.include_router(shop_entry_point.router)