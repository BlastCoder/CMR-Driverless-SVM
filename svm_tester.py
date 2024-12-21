import matplotlib.pyplot as plt
import numpy as np

# Define ator and ftom functions
def ator(degree):
    return np.radians(degree)  # Convert degrees to radians

def ftom(feet):
    return feet * 0.3048  # Convert feet to meters

# Define blue_list, yellow_list, and additional_points
"""
original points:

blue_list = [
    [-4, 0], [-4, 2], [-4, 4], [-4, 6], [-4, 8], [-4, 10], [-4, 12], [-4, 14], [-4, 16], [-4, 18], 
    [-4, 20], [-4, 22], [-4 - 2 + 2 * np.cos(ator(30)), 22 + 2 * np.sin(ator(30))], 
    [-4 - 2 + 2 * np.cos(ator(60)), 22 + 2 * np.sin(ator(60))], [-6, 24], [-8, 24], [-10, 24], 
    [-12, 24], [-14, 24], [-14 + 2 * np.cos(ator(120)), 24 - 2 + 2 * np.sin(ator(120))], 
    [-14 + 2 * np.cos(ator(150)), 24 - 2 + 2 * np.sin(ator(150))], [-16, 22], [-16 + ftom(4), 20], 
    [-16 + ftom(6), 18], [-16 + ftom(2), 16], [-16 - ftom(2), 14], [-16 - ftom(6), 12], 
    [-16 - ftom(4), 10], [-16, 8], [-16, 6], [-16, 4], 
    [-16 + 2 - 2 * np.cos(ator(30)), 4 - 2 * np.sin(ator(30))], [-14, 2], [-12, 2], [-10, 2], 
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
"""

blue_list = [
    [-4, 0], [-4, 2], [-4, 4], [-4, 6], [-4, 8], [-4, 10], [-4, 12], [-4, 14], [-4, 16], [-4, 18], 
    [-4, 20], [-4, 22]
]

yellow_list = [
    [0, 0], [0, 2], [0, 4], [0, 6], [0, 8], [0, 10], [0, 12], [0, 14], [0, 16], [0, 18], 
    [0, 20], [0, 22]
]

additional_points = [
    (-1.5, -0.4),
    (-1.5, -4.80171e-15),
    (-1.5, 0.4),
    (-1.5, 0.9),
    (-1.5, 1.3),
    (-1.5, 1.7),
    (-1.5, 2.1),
    (-1.5, 2.5),
    (-1.5, 2.9),
    (-1.5, 3.3),
    (-1.5, 3.7),
    (-1.5, 4.1),
    (-1.5, 4.6),
    (-1.5, 5.1),
    (-1.5, 5.6),
    (-1.5, 6.1),
    (-1.5, 6.6),
    (-1.5, 7.1),
    (-1.5, 7.6),
    (-1.5, 8.1),
    (-1.5, 8.6),
    (-1.5, 9.1),
    (-1.5, 9.6),
    (-1.5, 10.1),
    (-1.5, 10.6),
    (-1.5, 11.1),
    (-1.5, 11.6),
    (-1.5, 12.1),
    (-1.5, 12.6),
    (-1.5, 13.1),
    (-1.5, 13.6),
    (-1.5, 14.1),
    (-1.5, 14.6),
    (-1.5, 15.1),
    (-1.5, 15.6),
    (-1.5, 16.1),
    (-1.5, 16.5),
]

points2 = [
    (-2.0, -1.7),
    (-2.0, -1.3),
    (-2.0, -0.9),
    (-2.0, -0.5),
    (-2.0, -0.1),
    (-2.0, 0.3),
    (-2.0, 0.7),
    (-2.0, 1.1),
    (-2.0, 1.5),
    (-2.0, 1.9),
    (-2.0, 2.3),
    (-2.0, 2.7),
    (-2.0, 3.1),
    (-2.0, 3.5),
    (-2.0, 3.9),
    (-2.0, 4.3),
    (-2.0, 4.7),
    (-2.0, 5.1),
    (-2.0, 5.5),
    (-2.0, 5.9),
    (-2.0, 6.3),
    (-2.0, 6.7),
    (-2.0, 7.1),
    (-2.0, 7.5),
    (-2.0, 7.9),
    (-2.0, 8.3),
    (-2.0, 8.7),
    (-2.0, 9.1),
    (-2.0, 9.5),
    (-2.0, 9.9),
    (-2.0, 10.3),
    (-2.0, 10.7),
    (-2.0, 11.1),
    (-2.0, 11.5),
    (-2.0, 11.9),
    (-2.0, 12.3),
    (-2.0, 12.7),
    (-2.0, 13.1),
    (-2.0, 13.5),
    (-2.0, 13.9),
    (-2.0, 14.3),
    (-2.0, 14.7),
    (-2.0, 15.1)
]


# Convert lists to arrays for easier plotting
blue_points = np.array(blue_list)
yellow_points = np.array(yellow_list)
additional_points = np.array(additional_points)
points2 = np.array(points2)

# Extract x and y coordinates
blue_x, blue_y = blue_points[:, 0], blue_points[:, 1]
yellow_x, yellow_y = yellow_points[:, 0], yellow_points[:, 1]
additional_x, additional_y = additional_points[:, 0], additional_points[:, 1]
two_x, two_y = points2[:, 0], points2[:, 1]

# Plot the points
plt.figure(figsize=(10, 10))
plt.scatter(blue_x, blue_y, color='blue', label='Blue Cones')
plt.scatter(yellow_x, yellow_y, color='yellow', label='Yellow Cones')
plt.scatter(additional_x, additional_y, color='red', label='C++ Points')
# plt.scatter(two_x, two_y, color='green', label='Python Points')

# Add labels and legend
plt.title('Blue, Yellow, and Midline Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Show the plot
plt.show()
