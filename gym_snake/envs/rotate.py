import math

def rotate(X, Y, theta):
    x = X*math.cos(theta) - Y*math.sin(theta)
    y = X*math.sin(theta) + Y*math.cos(theta)
    return x, y
