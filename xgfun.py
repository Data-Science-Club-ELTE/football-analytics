import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch

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


# =========================
# BATCH VERSION (DATAFRAME)
# =========================
def calculate_xg_df(df, x_col='x', y_col='y', normalize=True):
    """
    Add xG column to a dataframe
    """

    x_vals = df[x_col].values.copy()
    y_vals = df[y_col].values.copy()

    if normalize:
        x_vals = 100 - x_vals
        y_vals = 100 - y_vals

        # overwrite dataframe columns
        df[x_col] = x_vals
        df[y_col] = y_vals

    # Distance
    distance = np.sqrt((GOAL_X - x_vals)**2 + (GOAL_Y - y_vals)**2)

    # Distances to posts
    dist_left = np.sqrt((GOAL_X - x_vals)**2 + (LEFT_POST_Y - y_vals)**2)
    dist_right = np.sqrt((GOAL_X - x_vals)**2 + (RIGHT_POST_Y - y_vals)**2)

    goal_width = RIGHT_POST_Y - LEFT_POST_Y

    cos_angle = (
        dist_left**2 + dist_right**2 - goal_width**2
    ) / (2 * dist_left * dist_right)

    cos_angle = np.clip(cos_angle, -1, 1)

    angle = np.arccos(cos_angle)

    # Logistic regression
    z = INTERCEPT + B_DISTANCE * distance + B_ANGLE * angle
    xg = logistic(z)

    df = df.copy()
    df['xg'] = xg

    return df


# =========================
# Viz
# =========================
def plot_xg(
    df,
    x_col='x',
    y_col='y',
    xg_col='xg',
    title='Shot Map (xG)',
    cmap='viridis',
    alpha=0.7,
    figsize=(10, 7)
):
    """
    Plot shot locations colored by xG on an Opta pitch

    Parameters:
        df : pandas DataFrame
        x_col, y_col : column names for coordinates
        xg_col : column name for xG values
        title : plot title
        cmap : color palette
        alpha : transparency
        figsize : figure size
    """

    # Create pitch
    pitch = Pitch(
        pitch_type='opta',
        pitch_color='#aabb97',
        line_color='white',
        stripe_color='#c2d59d',
        stripe=True,
        positional=True,
        positional_color='white'
    )

    fig, ax = pitch.draw(figsize=figsize)

    # Scatter plot
    scatter = sns.scatterplot(
        x=x_col,
        y=y_col,
        data=df,
        hue=xg_col,
        palette=cmap,
        alpha=alpha,
        ax=ax,
        legend=True
    )

    # Improve legend
    norm = plt.Normalize(df[xg_col].min(), df[xg_col].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('xG')

    ax.set_title(title)

    return fig, ax
