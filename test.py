import torch
import os
import json
import argparse
import cv2
import numpy as np
from network import Network

if __name__ == "__main__":

    config = json.load(open("config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Network()
    model.load_state_dict(
        torch.load(
            os.path.join(config["save_model_path"], "best_model.pth"),
            map_location=device,
        )
    )
    model.eval().to(device).float()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to the image")
    parser.add_argument(
        "-l", "--label", required=True, type=int, help="Actual label of the image"
    )
    args = parser.parse_args()

    img = cv2.imread(args.path)
    img = cv2.resize(img, (400, 400))
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img).to(device).float()
    pred = model(img)

    print("Raw Predictions:", pred)
    print("Softmax Probabilities:", torch.softmax(pred, 1))
    predicted_class = torch.argmax(torch.softmax(pred, 1), 1)
    print("Predicted Class:", predicted_class)
    print("Is Prediction Correct:", predicted_class.item() == args.label)
