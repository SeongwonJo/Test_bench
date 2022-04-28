import requests


def send_alarm_to_slack(msg):
    url = "https://hooks.slack.com/services/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"

    payload = {"text" : msg }

    requests.post(url, json=payload)


if __name__ == "__main__":
    send_alarm_to_slack("test")