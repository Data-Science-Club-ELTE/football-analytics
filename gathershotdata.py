from SofaScore import get_season_by_id
import ScraperFC as sfc
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import math
from tqdm import tqdm



PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32

def main():
    ss = sfc.Sofascore()
    TOURNAMENT = "HUN_NB1"  # Hungarian NB I (Fizz Liga)
    SEASON = "2025_2026"
    tournament_dict = {
        "HUN_NB1" : 187
    }

    season_dict = {
        "2024_2025" : "24/25",
        "2025_2026" : "25/26",
        "2023_2024" : "23/24"
    }

    

    match_ids = get_season_by_id(tournament_dict[TOURNAMENT], season_dict[SEASON])
    # match_ids = match_ids[0:4]
    OUTPUT_PATH = "Output"
    df = pd.DataFrame()

    for i in tqdm(match_ids):
        shots_data = ss.scrape_match_shots(i)
        needed_data = shots_data[["shotType", "situation", "playerCoordinates", "bodyPart"]]
        needed_data["goal"] = needed_data["shotType"].apply(
            lambda x: "goal" if str(x).lower() == "goal" else "no goal"
        )
        needed_data.drop("shotType", axis=1, inplace=True)
        needed_data[["x", "y"]] = needed_data["playerCoordinates"].apply(
            lambda d: pd.Series({"x": d.get("x"), "y": d.get("y")}) if isinstance(d, dict) else pd.Series({"x": None, "y": None})
        )

        needed_data["shot_angle_deg"] = [
            shooting_angle_xg(x, y)
            for x, y in zip(needed_data["x"], needed_data["y"])
        ]
        needed_data.drop("playerCoordinates", axis=1, inplace=True)
        df = pd.concat([df, needed_data])
    df.reset_index(drop=True, inplace=True)
    df.to_excel(f"{OUTPUT_PATH}/League_goals_{TOURNAMENT}_{SEASON}.xlsx")



def shooting_angle_xg(x, y, goal_x=0.0, goal_y=50.0,
                      pitch_width_m=PITCH_WIDTH_M, goal_width_m=GOAL_WIDTH_M):
    # goal width in SofaScore y-units (0..100 corresponds to pitch width)
    goal_width_y = 100.0 * goal_width_m / pitch_width_m
    half = goal_width_y / 2.0

    # goalposts (goal on x = 0, centered at y=50)
    post1 = (goal_x, goal_y - half)  # upper post
    post2 = (goal_x, goal_y + half)  # lower post

    # vectors from shot point to posts
    v1x, v1y = post1[0] - x, post1[1] - y
    v2x, v2y = post2[0] - x, post2[1] - y

    # dot and norms
    dot = v1x * v2x + v1y * v2y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)

    if n1 == 0 or n2 == 0:
        return 0.0  # shot exactly at a post (degenerate)

    # numerical safety
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    angle_rad = math.acos(cosang)
    return math.degrees(angle_rad)


if __name__ == "__main__":
    main()