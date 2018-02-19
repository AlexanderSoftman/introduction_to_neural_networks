import numpy as np
import csv
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def make_iteration(W_line, b_line):
    with open('data.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        dot_list = []
        for dot_pars in spamreader:
            new_W_line, new_b_line = perceptronStep(dot_pars, W_line, b_line)
            print(dot_pars)
            dot_list.append(dot_pars)
            W_line = new_W_line
            b_line = new_b_line
            if (dot_pars[2] == '1'):
                plt.plot([dot_pars[0]], [dot_pars[1]], 'ro')
            elif (dot_pars[2] == '0'):
                plt.plot([dot_pars[0]], [dot_pars[1]], 'bo')
        dot_list = sorted(
            dot_list,
            key=lambda value: float(value[0]))
        print("dot_list: %s" % (dot_list,))
        print("min x: %s, max_x: %s" % (float(dot_list[0][0]), float(dot_list[-1][0])))
        # create plot
        x1 = [float(dot_list[0][0]), float(dot_list[-1][0])]
        y1 = [
            (W_line[0] * x1[0] + b_line) / (-W_line[1]),
            (W_line[0] * x1[1] + b_line) / (-W_line[1])]
        plt.plot(x1, y1, marker='o')
        plt.axis('auto')
        plt.show()
        return (W_line, b_line)


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction(
        X[0] * W[0] + X[1] * W[1] + b)


# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(dot_pars, W_line, b_line, learn_rate=0.01):
    dot_pars_float = [
        float(dot_pars[0]),
        float(dot_pars[1]),
        float(dot_pars[2])]
    if (prediction(
            dot_pars_float, W_line, b_line) == 0) and (
                dot_pars_float[2] == 1):
        W_line[0] = W_line[0] + learn_rate * dot_pars_float[0]
        W_line[1] = W_line[1] + learn_rate * dot_pars_float[1]
        b_line = b_line + learn_rate
    elif (prediction(
            dot_pars_float, W_line, b_line) == 1) and (
                dot_pars_float[2] == 0):
        W_line[0] = W_line[0] - learn_rate * dot_pars_float[0]
        W_line[1] = W_line[1] - learn_rate * dot_pars_float[1]
        b_line = b_line - learn_rate
        # if (pred)
    return W_line, b_line


def main():
    iteration_count = 50
    # initial perceptron parameters
    W_line = [1, -1]
    b_line = 0
    for i in range(iteration_count):
        W_line, b_line = make_iteration(W_line, b_line)


if __name__ == '__main__':
    main()
