from sympy import *


if __name__ == '__main__':
    w = symbols('w')
    x = symbols('x')
    y = symbols('y')

    f = w * x

    cost = 1/2*(y-f)**2
    print ('Linear Regression:', diff(cost, w))
    # Linear Regression: x*(-y + (-y + 1)*exp(w*x))/(exp(w*x) + 1)
    # Equals to: x * (y - wx)

    sigmoid = 1 / (1 + exp(-f))
    cost = -y*log(sigmoid)-(1-y)*log(1-sigmoid)
    print ('Logistic Regression:', simplify(diff(cost, w)))
    # Logistic Regression: x*(-y + (-y + 1)*exp(w*x))/(exp(w*x) + 1)
    # Equals to: x * (y - sigmoid(wx))
