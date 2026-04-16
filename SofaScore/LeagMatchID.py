import requests

BASE = "https://www.sofascore.com/api/v1"

def get_season_by_id(UT_ID, SEASON_LABEL):

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    def get_json(url):
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    # --- get season id by label ---
    seasons = get_json(f"{BASE}/unique-tournament/{UT_ID}/seasons")["seasons"]
    season_id = None
    for s in seasons:
        name = (s.get("name") or "").lower()
        year = (s.get("year") or "").lower()
        if SEASON_LABEL.lower() in name or SEASON_LABEL.lower() in year:
            season_id = s["id"]
            break

    if season_id is None:
        raise ValueError(f"Season '{SEASON_LABEL}' not found")

    # --- collect finished matches ---
    event_ids = set()
    round_no = 1
    empty_rounds = 0

    while empty_rounds < 3:
        try:
            data = get_json(
                f"{BASE}/unique-tournament/{UT_ID}/season/{season_id}/events/round/{round_no}"
            )
        except:
            break
        events = data.get("events", [])

        if not events:
            empty_rounds += 1
            round_no += 1
            continue

        empty_rounds = 0

        for e in events:
            if e.get("status", {}).get("type") == "finished":
                event_ids.add(e["id"])

        round_no += 1

    
    event_ids = list(event_ids)
    print(f"\nFinished matches in {SEASON_LABEL}: {len(event_ids)}")
    return event_ids

