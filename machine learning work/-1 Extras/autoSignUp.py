from secret import *
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import requests
import schedule
import os
import time
from typing import Callable
import platform
import datetime

AUTHORIZATION_URL = "https://ion.tjhsst.edu/oauth/authorize"
TOKEN_URL = "https://ion.tjhsst.edu/oauth/token"
SCHEDULE_URL = "https://ion.tjhsst.edu/api/schedule"
BLOCK_URL = "https://ion.tjhsst.edu/api/blocks"
SIGNUP_URL = "https://ion.tjhsst.edu/api/signups/user"
PATH = "C:\Program Files (x86)\chromedriver.exe"


def get_code() -> str:

    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = Chrome(PATH, options=chrome_options)

    data = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "scope": "read+write",
    }

    url = AUTHORIZATION_URL + "?" + \
        "&".join(f"{k}={v}" for k, v in data.items())

    driver.get(url)

    username_box = driver.find_element(By.ID, "id_username")
    password_box = driver.find_element(By.ID, "id_password")

    username_box.send_keys(USERNAME)
    password_box.send_keys(PASSWORD + Keys.RETURN)

    authorize_xpath = '//input[@type="submit" and @value="Authorize"]'

    authorize_btn = driver.find_element(By.XPATH, authorize_xpath)
    authorize_btn.click()

    code = driver.current_url.split("=")[1].split("&")[0]

    driver.quit()

    return code


def get_token(code: str = "") -> dict:
    if code == "":
        code = get_code()

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }

    r = requests.post(TOKEN_URL, data=data)

    return r.json()["access_token"]

def get_date(**kwargs) -> str:
    today_date = datetime.date.today().strftime("%Y-%m-%d")
    params = {"start_date": today_date}
    r = requests.get(BLOCK_URL, params=params, **kwargs)
    dates = []
    for block in r.json()["results"]:
        date = block["date"]
        if date not in dates: dates.append(date)

    print("Select a date: ")
    for i, date in enumerate(dates):
        print(f"{i}: {date}")

    user_input = input()

    assert user_input.isdigit()

    user_input = int(user_input)

    assert user_input in range(len(dates))

    return dates[user_input]


def get_blocks(date: str, **kwargs) -> list[str]:
    params = {"date": date}
    r = requests.get(BLOCK_URL, params=params, **kwargs)
    blocks = []
    for block in r.json()["results"]:
        blocks.append(str(block["id"]))

    return blocks


def get_activities(block_id: str, **kwargs) -> dict[str, str]:

    r = requests.get(BLOCK_URL + "/" + block_id, **kwargs)

    activities = {"0": "NA"}
    for id, activity in r.json()["activities"].items():
        activities[id] = activity["name"]

    return activities


def get_activity(activities: dict[str, str]) -> str:

    for id, activity in activities.items():
        print(f"{id}: {activity}")

    user_input = input()

    assert user_input in activities.keys() or user_input == "NA"

    return user_input


def schedule_run(func: Callable, **kwargs) -> None:
    # schedule.every().second.do(lambda: func(**kwargs))
    schedule.every().day.at("00:01").do(lambda: func(**kwargs))


def signup(blocks: list[str], activities: list[str], **kwargs) -> None:
    for block_id, activity_id in zip(blocks, activities):
        if activity_id == "0":
            continue
        data = {
            "block": block_id,
            "activity": activity_id,
            "scheduled_activity": "162371",
            "use_scheduled_activity": "false",
            "force": "false"
        }
        r = requests.post(SIGNUP_URL, data=data, **kwargs)
        print(r.json())

    return schedule.CancelJob


def await_run() -> None:

    while len(schedule.get_jobs()) != 0:
        schedule.run_pending()
        time.sleep(1)

    # system = platform.system()

    # if system == "Windows":
    #     os.system("shutdown -s -t 10")
    # elif system == "Linux":
    #     os.system("shutdown -h")


def main() -> None:
    code = get_code()
    token = get_token(code)

    headers = {"Authorization": f"Bearer {token}"}

    date = get_date(headers=headers)
    blocks = get_blocks(date, headers=headers)
    selected_activities = []
    for block_id in blocks:

        activities = get_activities(block_id, headers=headers)

        activity = get_activity(activities)

        selected_activities.append(activity)

    schedule_run(signup, blocks=blocks,
                 activities=selected_activities, headers=headers)
    print("-" * 50)
    print(f"date: {date}")
    for activity in selected_activities:
        print(f"activity: {activity}")
    await_run()


if __name__ == "__main__":
    main()
