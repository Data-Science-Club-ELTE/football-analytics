import numpy as np

# =========================
# MODEL COEFFICIENTS
# =========================
INTERCEPT = -2.0764063859657456
B_DISTANCE = -0.04436557739712295
B_ANGLE = 1.1330379672764233 


# =========================
# CONSTANTS (OPTA PITCH)
# =========================
GOAL_X = 100
GOAL_Y = 50

LEFT_POST_Y = 44
RIGHT_POST_Y = 56


# =========================
# FEATURE ENGINEERING
# =========================
def compute_features(x, y):
    """
    Compute distance + angle from coordinates
    """

    # Distance
    distance = np.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2)

    # Distances to posts
    dist_left = np.sqrt((GOAL_X - x)**2 + (LEFT_POST_Y - y)**2)
    dist_right = np.sqrt((GOAL_X - x)**2 + (RIGHT_POST_Y - y)**2)

    # Goal width
    goal_width = RIGHT_POST_Y - LEFT_POST_Y

    # Angle
    angle = np.arccos(
        (dist_left**2 + dist_right**2 - goal_width**2) /
        (2 * dist_left * dist_right)
    )

    return distance, angle


def normalize_coordinates(x, y):
    # Flip to attacking right
    if x < 50:
        x = 100 - x
    # Vertical flip
        y = 100 - y
    
    return x, y


# =========================
# LOGISTIC FUNCTION
# =========================
def logistic(z):
    return 1 / (1 + np.exp(-z))


# =========================
# SINGLE SHOT xG
# =========================
def calculate_xg(x, y):
    """
    Calculate xG for a single shot
    """

    x, y = normalize_coordinates(x, y)
    distance, angle = compute_features(x, y)

    X = np.array([[distance, angle]])

    z = INTERCEPT + B_DISTANCE * distance + B_ANGLE * angle

    return logistic(z)

