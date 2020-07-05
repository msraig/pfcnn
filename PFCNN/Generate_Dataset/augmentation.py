from scipy.linalg import expm, norm
import math
import numpy as np

def M(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

def rotated_points(points, axis, theta):
    return np.dot(points, M(axis, theta).T)

class Augmentor:
    def __init__(self, data, A_num):
        self.augment_num = A_num - 1
        self.original_data = data
        self.iter = 0
        self.axis = np.random.uniform(-1, 1, size=(self.augment_num ,3))
        self.theta = np.random.uniform(-math.pi, math.pi, size=(self.augment_num))
    def get_next(self):
        if(self.iter==0):
            self.iter = self.iter + 1
            return self.original_data
        else:
            self.iter = (self.iter + 1) % self.augment_num
            return rotated_points(self.original_data, self.axis[self.iter-1], self.theta[self.iter-1])
