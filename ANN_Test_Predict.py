import unittest
from ANN import ANN
import numpy as np


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.myANN = ANN.fromFile("testingANN.dat")
# =======================================================  FORWARD PROPOGATION TESTS for TO-DO #0
    def test_1_first_activation(self):
        print("test 1")
        self.myANN.predict(np.array((0.8, 0.6, 0.4, 0.2, 0)))
        print(self.myANN.y_hats_for_layers[0])
        expected1 = np.array([0.68997448, 0.64565631, 0.59868766, 0.549834,   0.5])
        np.testing.assert_almost_equal(actual=self.myANN.y_hats_for_layers[0], desired=expected1, decimal=5)

    def test_2_a_and_y_hats(self):
        print("test 2")
        self.myANN.predict(np.array((1.0, -0.5, 0.0, -0.5, -1.0)))
        expected_as = [np.array([1., -0.5, 0., -0.5, -1.]),
                       np.array([1.67219403, 1.98473154, 1.39369625, 1.54398201, 1.94811212, 2.06485024, 2.02812228,
                                 1.60509281]),
                       np.array([4.32923312, 3.6118534, 2.78584314, 2.87044873, 2.55395664, 3.23304541]),
                       np.array([3.73886022, 2.91403626, 4.59257246])]
        expected_y_hats = [np.array([0.73105858, 0.37754067, 0.50000000, 0.37754067, 0.26894142]),
                           np.array([0.84186812, 0.87918464, 0.80118167, 0.82404285, 0.87524064, 0.88743957, 0.88371826,
                                     0.83272897]),
                           np.array([0.98699374, 0.97370817, 0.94190600, 0.94636613, 0.92783888, 0.96205907]),
                           np.array([0.97677122, 0.94853595, 0.98997475])]

        for i in range(len(expected_as)):
            np.testing.assert_almost_equal(actual=self.myANN.a_values_for_layers[i], desired=expected_as[i], decimal=5)
        for i in range(len(expected_y_hats)):
            np.testing.assert_almost_equal(actual=self.myANN.y_hats_for_layers[i],
                                           desired=expected_y_hats[i],
                                           decimal=5)

    def test_3_prediction(self):
        print("test 3")
        output = self.myANN.predict(np.array((-0.5, -0.25, 0, 0.25, 0.5)))
        print(f"{output=}")
        expected3 = np.array((0.97687363, 0.94875073, 0.99003326))
        np.testing.assert_almost_equal(actual=output, desired=expected3, decimal=5)


if __name__ == '__main__':
    unittest.main()
