from src.computer_vision.controller import ComputerVisionController
from src.sample import start_sample


def tensorflow_example():
    start_sample()


def test():
    cv = ComputerVisionController()
    cv.train()
    result_dict = cv.predict('/home/viti/Documents/UEB/Semestre9/SistemasInteligentes/ComputerVision/validate/',
                             'validate_reloj.png')
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(result_dict["class"],
                                                                                        100 * result_dict["percent"]))


def main():
    test()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
