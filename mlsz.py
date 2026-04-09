import requests
from bs4 import BeautifulSoup
import pandas as pd


results = []
#url = "https://adatbank.mlsz.hu/league/65/5/32165/15.html"  # replace with your site
for i in range(18):
    url = f"https://adatbank.mlsz.hu/league/65/0/31362/{i+1}.html"
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    box = soup.find("div", class_=["box", "box-1"])
    schedules = box.find_all("div", class_="schedule") if box else []
    for sched in schedules:
        # Home team
        home_team = sched.find("div", class_="home_team")
        home_team_name = home_team.get_text(strip=True) if home_team else None

        # Away team
        away_team = sched.find("div", class_="away_team")
        away_team_name = away_team.get_text(strip=True) if away_team else None

        # Result
        result_span = sched.find("span", class_="schedule-points")
        result = result_span.get_text(strip=True) if result_span else None

        results.append({
            "home_team": home_team_name,
            "away_team": away_team_name,
            "result": result
        })

df = pd.DataFrame(results)
df[["HomeGoal", "-", "AwayGoal"]] = df["result"].str.split(" ", expand=True)

df.drop(columns=["-", "result"], inplace=True)
df["HomeGoal"] = df["HomeGoal"].astype(int)
df["AwayGoal"] = df["AwayGoal"].astype(int)
print(df)
print(df.dtypes)
df.to_excel("FIZZ2526.xlsx")