"""
Docstring for xgmodel.


What is xG?
Very simply, xG (or expected goals) is the probability that a shot will result in a goal based on the characteristics of that shot and the events leading up to it.

Source: https://fbref.com/en/expected-goals-model-explained/

1. Gyűjtsük ki az összes összes lövést mondjuk az elmúlt 2 évből a magyar bajnokságból.
2. Szedjük ki a paramétereiket
 -- shottype - this will be the y variable, eithor goal or not
 -- player Coordinates - x - X coordinate of the player
 -- player Coordinates - y - Y coordinate of the player
 -- shot angle - Angle of the shot based on the player cordinates x and y
 -- situation - How to player got the pass - It needs to be changed to numerical
 -- bodyPart - Which Bodypart does the player used - It needs to be changed to numerical

3. Miután összegyűjtöttök a lövéseket szétszedjük őket train-test-split-re
4. Felépítünk egy modelt, ami az alapján, hogy mik az input paraméterek, kiköp egy valószínűséget, hogy az gól
"""


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Models import XGmodel
from sklearn.metrics import brier_score_loss
import seaborn as sns
import matplotlib.pyplot as plt





def main():
    OUTPUT_PATH = "Output"
    TOURNAMENT = "HUN_NB1"  # Hungarian NB I (Fizz Liga)
    SEASON = "2024_2025"

    f"{OUTPUT_PATH}/League_goals_{TOURNAMENT}_{SEASON}.xlsx"
    le_situation = LabelEncoder()
    le_body = LabelEncoder()

    goal_map = {
        "goal": 1,
        "no goal" : 0
    }

    shot_data_df = pd.read_excel(r"Output\League_goals_HUN_NB1_2024_2025.xlsx", index_col=0)

    shot_data_df["situation_enc"] = le_situation.fit_transform(shot_data_df["situation"])
    shot_data_df["bodyPart_enc"] = le_body.fit_transform(shot_data_df["bodyPart"])
    shot_data_df["goal_enc"] = shot_data_df["goal"].map(goal_map)

    

    shot_data_df.drop(["situation", "bodyPart", "goal"] , axis=1, inplace=True)
    sns.heatmap(shot_data_df.corr())
    plt.show()
    mapping_for_situation = dict(zip(le_situation.classes_, le_situation.transform(le_situation.classes_)))
    mapping_for_body = dict(zip(le_body.classes_, le_body.transform(le_body.classes_)))

    X = shot_data_df[["x", "y", "shot_angle_deg", "situation_enc", "bodyPart_enc"]]
    Y = shot_data_df["goal_enc"]


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    

    
    model = XGmodel(model_type='Deep Learning', num_iters=30)
    model.fit(X_train, y_train)
    predictions, probabilities = model.predict(X_test)

    brier_loss = round(brier_score_loss(y_test, probabilities), 3)
    print(f"For the DL: Brier Score Loss: {brier_loss}")

if __name__ == "__main__":
    main()