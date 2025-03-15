
import requests
import sys
sys.path.append("../")
from task_3.tokenn import TOKEN

result = requests.post(
        "http://149.156.182.9:6060/task-1/submit",
        headers={"token": TOKEN},
        files={
            "csv_file": ("submission.csv", open("./data/pub_submission.csv", "rb"))
        }
    )

print(result.status_code, result.text)