import requests
import pandas as pd
from datetime import date
import time

def get_game_pks(start_date, end_date):
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
    }
    headers = {
        "User-Agent": "MLB-Pitch-Sequence-Modelling/1.0"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()

            game_pks = []

            for d in data.get("dates", []):
                for g in d.get("games", []):
                    game_pks.append(g["gamePk"])

            return game_pks
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\nError fetching game list (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed to fetch game list after {max_retries} attempts")
                raise

def get_pitch_feed(game_pk):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    headers = {
        "User-Agent": "MLB-Pitch-Sequence-Modelling/1.0"
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract game_date from the API response
            game_date = data.get("gameData", {}).get("datetime", {}).get("officialDate")

            all_plays = data["liveData"]["plays"]["allPlays"]

            pitch_data = []

            for play in all_plays:
                ab_idx = play.get("atBatIndex")
                inning = play.get("about", {}).get("inning")
                matchup = play.get("matchup", {})
                pitcher_id = matchup.get("pitcher", {}).get("id")
                batter_id = matchup.get("batter", {}).get("id")
                pitcher_hand = (
                    matchup.get("pitchHand", {})
                        .get("code")
                )
                batter_hand = (
                    matchup.get("batSide", {})
                        .get("code")
                )

                half_inning = play.get("about", {}).get("halfInning")
                pitcher_is_home = False

                if half_inning == "top":
                    pitcher_is_home = True

                events = play.get("playEvents", [])
                pitch_num_in_pa = 0

                for ev in events:
                    if not ev.get("isPitch"):
                        continue

                    pitch_num_in_pa += 1
                    count = ev.get("count", {})
                    balls_after_pitch = count.get("balls")
                    strikes_after_pitch = count.get("strikes")

                    baseRunners = ev.get("baseRunners", [])

                    #runner_on_1B = False
                    #runner_on_2B = False
                    #runner_on_3B = False

                    #for runner in baseRunners:
                        #if play.get("runnerOn1b") is not None:
                            #runner_on_1B = True
                        #elif play.get("runnerOn2b") is not None:
                            #runner_on_2B = True
                        #elif play.get("runnerOn3b") is not None:
                            #runner_on_3B = True

                    details = ev.get("details", {})
                    pitch_type = details.get("type", {}).get("code")

                    pitch_data.append({
                        "game_date": game_date,
                        "game_pk": game_pk,
                        "ab_idx": ab_idx,
                        "pitch_num_in_pa": pitch_num_in_pa,
                        "inning": inning,
                        "pitcher_id": pitcher_id,
                        "batter_id": batter_id,
                        "pitcher_hand": pitcher_hand,
                        "batter_hand": batter_hand,
                        "pitcher_is_home": pitcher_is_home,
                        "balls_after_pitch": balls_after_pitch,
                        "strikes_after_pitch": strikes_after_pitch,
                        "pitch_type": pitch_type
                    })

            df = pd.DataFrame(pitch_data)
            return df, game_date

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                print(f"\nError fetching game {game_pk} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed to fetch game {game_pk} after {max_retries} attempts")
                raise


def print_game_dates(game_pks):
    for game_pk in game_pks:
        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {"gamePk": game_pk}

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        dates = data.get("dates", [])
        if not dates:
            print(f"{game_pk}: date not found")
            continue

        game_date = dates[0]["date"]
        print(f"{game_pk}: {game_date}")
