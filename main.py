from fastapi import FastAPI
from src.api.routes.routes import set_api_routes
from src.api.middlewares.cors_middleware import set_cors
import uvicorn

app = FastAPI()

# Middlewares
set_cors(app)


@app.get("/")
def read_root():
    return {"data": "Hello there!"}


# Routes
set_api_routes(app)


def init_api():
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)


def main():
    init_api()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
