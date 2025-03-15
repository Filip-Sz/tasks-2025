import requests
import torch
import argparse


TOKEN = "" # Your token here
URL = "149.156.182.9:6060/task-3/submit"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            prog='Model Submission',
            description='Submits the solution to task 3',
            epilog='')
    
    parser.add_argument('path')
    parser.add_argument('-n', '--name', default='resnet50')
    args = parser.parse_args()

    # Send the model to the server
    response = requests.post(
        URL,
        headers={
            "token": TOKEN,
            "model-name": args.name
        },
        files={
            "model_state_dict": open(args.path, "rb")
        }
    )

    # Should be 400, the clean accuracy is too low
    print(response.status_code, response.text)