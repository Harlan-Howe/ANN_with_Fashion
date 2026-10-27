import random
from ANN import ANN
from utils import mnist_reader
import numpy as np
import cv2

class FashionTester:

    def __init__(self):
        print ("Loading testing data.")
        self.x_test, self.y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
        print(f"Loaded {self.y_test.shape} examples.")
        print ("Loading ANN from file")
        self.myANN = ANN.fromFile("fashion 81.dat")
        print(f"Loaded ANN: {self.myANN.layer_sizes=}")


def test_ANN_stats():
    # TEXT ONLY: test NUM_TO_CHECK random items from test data and score how the ANN did.
    score = 0
    print("Predicted\tActual")
    NUM_TO_CHECK = 1000
    for count in range(NUM_TO_CHECK):
        i = random.randint(0, len(ft.y_test) - 1)
        float_image = ft.x_test[i] / 256.0
        output2 = ft.myANN.predict(float_image)
        # print(output2)
        if np.argmax(output2) == ft.y_test[i]:
            x_or_check = "\033[32m√\033[0m"  # green check
            score += 1
        else:
            x_or_check = "\033[31mX\033[0m"  # red X
        print(f"{np.argmax(output2)}\t\t{ft.y_test[i]}\t{x_or_check}")
    print(f"Scoring: Over {NUM_TO_CHECK} samples, the ANN was {100 * score / NUM_TO_CHECK:3.1f}% correct.")


def test_ANN_graphics():
    # GRAPHICS: select several (different) images at random and display what the computer predicted.
    CLASS_NAMES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                   "Ankle Boot")
    screen = np.zeros((800, 1000), dtype=np.uint8)
    NUM_ROWS = 20
    for i in range(80):
        j = random.randint(0, len(ft.y_test) - 1)
        screen[40 * (i % NUM_ROWS):40 * (i % NUM_ROWS) + 28, (i // NUM_ROWS) * 200 + 2:(i // NUM_ROWS) * 200 + 30] = \
        ft.x_test[j].reshape(
            (28, 28))
        cv2.putText(screen, text=CLASS_NAMES[np.argmax(ft.myANN.predict(ft.x_test[j]))],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    org=(50 + (i // NUM_ROWS) * 200, 40 * (i % NUM_ROWS) + 25),
                    color=255, thickness=1)
    cv2.imshow("output", screen)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ft = FashionTester()
    test_ANN_stats()
    test_ANN_graphics()