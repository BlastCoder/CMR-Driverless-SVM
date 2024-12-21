
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# pointsList is list [blue, yellow]

def augment_dataset(X, mult=4, var=0.5):
    '''duplicates a dataset by multiplier and adds random noise to points'''
    X = np.concatenate([X] * mult).astype(np.float64)
    X += np.random.randn(X.shape[0], X.shape[1]) * var

    return X

def supplement_cones(pointsList):
    '''does in place, adds cones around origin to ground an SVM classifier'''
    pointsList[0].append([-2, -1])
    pointsList[1].append([2, -1])

    return pointsList

def augment_cones(pointsList, mult=8, var=0.35):
    '''duplicates the cones by some multiplier and adds Gaussian noise with 
    some varaince'''
    blue = np.array(pointsList[0])
    yellow = np.array(pointsList[1])

    blue = augment_dataset(blue, mult=mult, var=var)
    yellow = augment_dataset(yellow, mult=mult, var=var)

    blue = blue.tolist()
    yellow = yellow.tolist()

    return [blue, yellow]

def augment_dataset_circle(X, deg=20, radius=2):
    # X is a np array
    '''for each sample in X, adds additional points on circle of specified radius
    where circle lies on the first two dimensions of sample'''
    DEG_TO_RAD = np.pi / 180
    radian = deg * DEG_TO_RAD
    angles = np.arange(0, 2 * np.pi, step=radian).astype(np.float64)

    N = X.shape[0]

    # create duplicate points
    num_angles = angles.shape[0]
    X_extra = np.concatenate([X] * num_angles).astype(np.float64)
    angles = np.repeat(angles, N).astype(np.float64)

    radius = float(radius)
    X_extra[:, 0] += radius * np.cos(angles)
    X_extra[:, 1] += radius * np.sin(angles)
    print(f"X: {X}")
    print(f"Type: {type(X)}")
    print(f"Shape: {np.shape(X)}")
    print(f"X_extra: {X_extra}")
    print(f"Type: {type(X_extra)}")
    print(f"Shape: {np.shape(X_extra)}")
    return np.concatenate([X, X_extra])

def augment_cones_circle(pointsList, deg=20, radius=2):
    '''for each cone in ones, adds additional cones of same color on circle
    around cone with specified radius, separated by degrees'''
    blue = np.array(pointsList[0])
    yellow = np.array(pointsList[1])

    print(f"blue: {blue}")
    print(f"Type: {type(blue)}")
    print(f"Shape: {np.shape(blue)}")

    blue = augment_dataset_circle(blue, deg=deg, radius=radius) 
    yellow = augment_dataset_circle(yellow, deg=deg, radius=radius) 
    
    blue = blue.tolist()
    yellow = yellow.tolist()

    return [blue, yellow]

def cones_to_xy(pointsList):
    '''Converts cones to a dataset representation (X, y) where y is vector
    of 0/1 labels where 0 corresponds to blue and 1 corresponds to yellow
    '''
    blue = pointsList[0]
    yellow = pointsList[1]
    blue_cones = np.array(blue)
    yellow_cones = np.array(yellow)
    features = []

    for point in blue:
        features.append(0)
    for point in yellow:
        features.append(1)

    features = np.array(features)
    data1 = np.vstack([blue_cones, yellow_cones]) 

    return data1, features

def get_spline_start_idx(points):
    '''gets index of point with lowest y-axis value in points'''
    # get points that are all the lowest
    min_y = np.min(points[:, 1])
    idxs = np.where(points[:, 1] == min_y)[0]

    # take the point that is closest to x = 0
    closest_x_idx = np.argmin(abs(points[idxs, 0]))
    return idxs[closest_x_idx]

def get_closest_point_idx(points, curr_point):
    '''gets index of point in points closest to curr_point and returns the dist'''
    assert(points.shape[1] == curr_point.shape[0])
    sq_dists = np.sum((points - curr_point) ** 2, axis=1)
    idx = np.argmin(sq_dists)
    return idx, np.sqrt(sq_dists[idx])

def sort_boundary_points(points, max_spline_length=17.5):
    '''sorts boundary points by starting from the lowest point and 
    iteratively takes closest point from iteration's current point
    takes approx: 7-8ms

    can additionallyn limit the number of points that are being ran on 
    '''

    # TODO: recalculating distances each iteration
    # might be better to calculate all pair-wise distances at start
    # and then iteratively removing from the dataset for each iteration

    # TODO: integrate spacing of 50cm here instead of repeating it
    # TODO: integrate maximum length of spline

    spline_length = 0
    points = np.array(points)
    sorted_points = []

    # start from the lowest point along the y-axis
    idx = get_spline_start_idx(points)
    curr_point = points[idx, :]
    rem_points = np.delete(points, idx, axis=0)

    # add starting point to sorted points
    sorted_points.append(curr_point)

    while rem_points.shape[0] > 0 and spline_length < max_spline_length:

        # find closest point to curr_point
        idx, d = get_closest_point_idx(rem_points, curr_point)
        spline_length += d

        # update iterates
        curr_point = rem_points[idx, :]
        rem_points = np.delete(rem_points, idx, axis=0)

        # add closest point to sorted points
        sorted_points.append(curr_point)

    return np.array(sorted_points)

def cones_to_midline(pointsList):

    blue_cones = np.array(pointsList[0])
    yellow_cones = np.array(pointsList[1])

    if len(blue_cones) == 0 and len(yellow_cones) == 0:
        return []
    
    # augment dataset to make it better for SVM training  
    # cones = supplement_cones(pointsList)
    cones = augment_cones_circle(pointsList, deg=10, radius=1.2) 

    X, y = cones_to_xy(cones)

    model = svm.SVC(kernel='poly', degree=3, C=10, coef0=1.0)
    model.fit(X, y)

    # TODO: prediction takes 20-30+ ms, need to figureC out how to optimize
    step = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                        np.arange(y_min, y_max, step))

    svm_input = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(svm_input)
    Z = Z.reshape(xx.shape)

    print(Z)

    # get top-left corner (TL) and bottom-right (BR) corner of Z
    Z_TL = Z[:-1, :-1]
    Z_BR = Z[1:, :-1]
    Z_C = Z[1:, 1:]
    XX_C = xx[1:, 1:]
    YY_C = yy[1:, 1:]
    idxs = np.where(np.logical_or(Z_C != Z_TL, Z_C != Z_BR))
    boundary_xx = XX_C[idxs].reshape((-1, 1))
    boundary_yy = YY_C[idxs].reshape((-1, 1))
    boundary_points = np.concatenate([boundary_xx, boundary_yy], axis=1)

    # sort the points in the order of a spline
    boundary_points = sort_boundary_points(boundary_points)

    # downsample the points
    downsampled = []
    accumulated_dist = 0
    for i in range(1, len(boundary_points)):
        p1 = boundary_points[i]
        p0 = boundary_points[i-1]
        curr_dist = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        accumulated_dist += curr_dist
        if np.abs(accumulated_dist - 0.5) < 0.1: # TODO: make this 50cm
            downsampled.append(p1)
            accumulated_dist = 0
        
        if accumulated_dist > 0.55:
            accumulated_dist = 0

    downsampled = np.array(list(downsampled))
    # print(downsampled)

    return downsampled

def ator(degree):
    return np.radians(degree)  # Convert degrees to radians

def ftom(feet):
    return feet * 0.3048  # Convert feet to meters

def main():
    blue_list = [
        [-4, 0], [-4, 2], [-4, 4], [-4, 6], [-4, 8], [-4, 10], [-4, 12], [-4, 14], [-4, 16], [-4, 18], 
        [-4, 20], [-4, 22], [-4 - 2 + 2 * np.cos(ator(30)), 22 + 2 * np.sin(ator(30))], 
        [-4 - 2 + 2 * np.cos(ator(60)), 22 + 2 * np.sin(ator(60))], [-6, 24], [-8, 24], [-10, 24], 
        [-12, 24], [-14, 24], [-14 + 2 * np.cos(ator(120)), 24 - 2 + 2 * np.sin(ator(120))], 
        [-14 + 2 * np.cos(ator(150)), 24 - 2 + 2 * np.sin(ator(150))], [-16, 22], [-16 + ftom(4), 20], 
        [-16 + ftom(6), 18], [-16 + ftom(2), 16], [-16 - ftom(2), 14], [-16 - ftom(6), 12], 
        [-16 - ftom(4), 10], [-16, 8], [-16, 6], [-16, 4], 
        [-16 + 2 - 2 * np.cos(ator(30)), 4 - 2 * np.sin(ator(30))], 
        [-16 + 2 - 2 * np.cos(ator(60)), 4 - 2 * np.sin(ator(90))], [-14, 2], [-12, 2], [-10, 2], 
        [-8, 2], [-6, 2], [-4, 2]
    ]

    yellow_list = [
        [0, 0], [0, 2], [0, 4], [0, 6], [0, 8], [0, 10], [0, 12], [0, 14], [0, 16], [0, 18], 
        [0, 20], [0, 22], [0 - 6 + 6 * np.cos(ator(30)), 22 + 6 * np.sin(ator(30))], 
        [0 - 6 + 6 * np.cos(ator(60)), 22 + 6 * np.sin(ator(60))], [-6, 28], [-8, 28], [-10, 28], 
        [-12, 28], [-14, 28], [-14 + 6 * np.cos(ator(120)), 28 - 6 + 6 * np.sin(ator(120))], 
        [-14 + 6 * np.cos(ator(150)), 28 - 6 + 6 * np.sin(ator(150))], [-20, 22], [-20 + ftom(4), 20], 
        [-20 + ftom(6), 18], [-20 + ftom(2), 16], [-20 - ftom(2), 14], [-20 - ftom(6), 12], 
        [-20 - ftom(4), 10], [-20, 8], [-20, 6], [-20, 4], 
        [-16 + 2 - 6 * np.cos(ator(30)), 4 - 6 * np.sin(ator(30))], 
        [-16 + 2 - 6 * np.cos(ator(60)), 4 - 6 * np.sin(ator(90))], [-14, -2], [-12, -2], [-10, -2], 
        [-8, -2], [-6, -2], [-4, -2]
    ]

    yellow_straight = [
        [0, 0],
        [0, 2],
        [0, 4],
        [0, 6],
        [0, 8],
        [0, 10],
        [0, 12],
        [0, 14],
        [0, 16],
        [0, 18],
        [0, 20],
        [0, 22]
    ]

    blue_straight = [
        [-4, 0],
        [-4, 2],
        [-4, 4],
        [-4, 6],
        [-4, 8],
        [-4, 10],
        [-4, 12],
        [-4, 14],
        [-4, 16],
        [-4, 18],
        [-4, 20],
        [-4, 22]
    ]

    pointsList = [blue_list, yellow_list]
    straightList = [blue_straight, yellow_straight]
    # ans = cones_to_midline(pointsList)
    straightAns = cones_to_midline(straightList)
    print("ans: \n")
    # print(ans)
    print("\n")
    print("straight ans: \n")
    print(straightAns)

if __name__ == "__main__":
    main()