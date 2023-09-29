import logging
import os
import os.path

import cv2
import gdown
import mxnet as mx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# Define the Google Drive URL of the file you want to download
fair_7_url = "https://drive.google.com/uc?id=113QMzQzkBDmYMs9LwzvD-jxEZdBQ5J4X"
fair_4_url = "https://drive.google.com/uc?id=1kXdAsqT8YiNYIMm8p5vQUvNFwhBbT4vQ"

fair_7_path = "res34_fair_align_multi_7_20190809.pt"
fair_4_path = "res34_fair_align_multi_4_20190809.pt"

GENDER_MAP = {
    "F": 0,
    "M": 1,
}


def image_shape_to_bbox(
    img_height: int, img_width: int, margin_ratio=0.1
) -> np.ndarray:
    """
    Deduces face bounding box from image shape (height, width)
    assuming the image is loose crop of the face.
    """
    x_center = img_width / 2
    y_center = img_height / 2

    bbox_height = img_height * (1 - margin_ratio)
    bbox_width = img_width * (1 - margin_ratio * 2)

    bbox_x1 = x_center - bbox_width / 2
    bbox_y1 = y_center - bbox_height / 2
    bbox_x2 = x_center + bbox_width / 2
    bbox_y2 = y_center + bbox_height / 2

    return np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2])


class MXFaceDataset(Dataset):
    """
    Mxnet RecordIO face dataset.
    """

    def __init__(self, root_dir: str, transforms=None, **kwargs) -> None:
        super(MXFaceDataset, self).__init__()
        self.transform = transforms
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, "test.rec")
        path_imgidx = os.path.join(root_dir, "test.idx")
        path_imglst = os.path.join(root_dir, "test.lst")
        items = [
            line.strip().split("\t") for line in open(path_imglst, "r")
        ]  # img_idx, 0, img_path

        self.img_idx_to_path = {int(item[0]): item[-1] for item in items}

        logging.info("loading recordio %s...", path_imgrec)
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        logging.info("loading recordio %s done", path_imgrec)
        self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        img_idx = self.imgidx[index]
        s = self.imgrec.read_idx(img_idx)
        header, sample = mx.recordio.unpack_img(s, cv2.IMREAD_UNCHANGED)
        # cv2.imwrite(f"test_{index:05d}.jpg", sample)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)

        return (
            sample,
            header.label[0],
            img_idx,
            self.img_idx_to_path[img_idx],
        )

    def __len__(self):
        return len(self.imgidx)


@torch.no_grad()
def predidct_age_gender_race(save_prediction_at, root_dir="cropped_faces/"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_fair_7 = torchvision.models.resnet34(pretrained=False)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    if not os.path.isfile(fair_7_path):
        gdown.download(fair_7_url, fair_7_path, quiet=False, fuzzy=True)
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.load_state_dict(torch.load(fair_7_path, map_location=device))
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=False)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    if not os.path.isfile(fair_4_path):
        gdown.download(fair_4_url, fair_4_path, quiet=False, fuzzy=True)
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.load_state_dict(torch.load(fair_4_path, map_location=device))
    model_fair_4.eval()

    trans = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    ds = MXFaceDataset(root_dir, trans)

    for index in tqdm(range(len(ds))):
        image, id_label, img_idx, img_name = ds[index]
        face_names.append(img_name)
        # if index > 10:
        #     break
        # reshape image to match model dimensions (1 batch size)
        image = image.view([1, 3, 224, 224])
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        outputs = model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)

        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    result = pd.DataFrame(
        [
            face_names,
            race_preds_fair,
            race_preds_fair_4,
            gender_preds_fair,
            age_preds_fair,
            race_scores_fair,
            race_scores_fair_4,
            gender_scores_fair,
            age_scores_fair,
        ]
    ).T
    result.columns = [
        "face_name_align",
        "race_preds_fair",
        "race_preds_fair_4",
        "gender_preds_fair",
        "age_preds_fair",
        "race_scores_fair",
        "race_scores_fair_4",
        "gender_scores_fair",
        "age_scores_fair",
    ]
    result.loc[result["race_preds_fair"] == 0, "race"] = "White"
    result.loc[result["race_preds_fair"] == 1, "race"] = "Black"
    result.loc[result["race_preds_fair"] == 2, "race"] = "Latino_Hispanic"
    result.loc[result["race_preds_fair"] == 3, "race"] = "East Asian"
    result.loc[result["race_preds_fair"] == 4, "race"] = "Southeast Asian"
    result.loc[result["race_preds_fair"] == 5, "race"] = "Indian"
    result.loc[result["race_preds_fair"] == 6, "race"] = "Middle Eastern"

    # race fair 4

    result.loc[result["race_preds_fair_4"] == 0, "race4"] = "White"
    result.loc[result["race_preds_fair_4"] == 1, "race4"] = "Black"
    result.loc[result["race_preds_fair_4"] == 2, "race4"] = "Asian"
    result.loc[result["race_preds_fair_4"] == 3, "race4"] = "Indian"

    # gender
    result.loc[result["gender_preds_fair"] == 0, "gender"] = "Male"
    result.loc[result["gender_preds_fair"] == 1, "gender"] = "Female"

    # age
    result.loc[result["age_preds_fair"] == 0, "age"] = "0-2"
    result.loc[result["age_preds_fair"] == 1, "age"] = "3-9"
    result.loc[result["age_preds_fair"] == 2, "age"] = "10-19"
    result.loc[result["age_preds_fair"] == 3, "age"] = "20-29"
    result.loc[result["age_preds_fair"] == 4, "age"] = "30-39"
    result.loc[result["age_preds_fair"] == 5, "age"] = "40-49"
    result.loc[result["age_preds_fair"] == 6, "age"] = "50-59"
    result.loc[result["age_preds_fair"] == 7, "age"] = "60-69"
    result.loc[result["age_preds_fair"] == 8, "age"] = "70+"

    result[
        [
            "face_name_align",
            "race",
            "race4",
            "gender",
            "age",
            "race_scores_fair",
            "race_scores_fair_4",
            "gender_scores_fair",
            "age_scores_fair",
        ]
    ].to_csv(save_prediction_at, index=False)

    print("saved results at ", save_prediction_at)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    root_dir = "/home/ubuntu/data/multilabel_5_1_cache_2/test/"
    # Please change test_outputs.csv to actual name of output csv.
    predidct_age_gender_race("../results/fairface_outputs.csv", root_dir)
