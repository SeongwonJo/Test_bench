import requests


def send_alarm_to_slack(msg):
    url = "slack web hook"

    payload = {"text" : msg }

    requests.post(url, json=payload)


if __name__ == "__main__":
    send_alarm_to_slack("test")
