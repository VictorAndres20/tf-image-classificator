import time

from src.computer_vision.controller import ComputerVisionController


def train_and_save():
    cv = ComputerVisionController()
    cv.train()
    cv.save()


def load_and_predict():
    cv = ComputerVisionController()
    cv.load()
    result_dict = cv.predict('/home/viti/Documents/UEB/Semestre9/SistemasInteligentes/ComputerVision/validate/',
                             'validate_reloj.png')
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(result_dict["class"],
                                                                                        100 * result_dict["percent"]))


def main():
    train_and_save()
    print("Trained")
    time.sleep(2)
    print("Loading and predicting")
    load_and_predict()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
