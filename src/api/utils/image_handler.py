import base64


def write_image(img_data: str, name: str):

    with open(name, "wb") as fh:
        fh.write(base64.decodebytes(img_data.encode()))
