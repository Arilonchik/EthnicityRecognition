from Face_detector import face_detector as fd
import cv2 as cv
from torchvision import models
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import os
import json


def get_classes(class_file):
    class_to_idx = json.load(open(class_file))

    idx_to_class = {v:k for k, v in class_to_idx.items()}
    classes = list(class_to_idx.keys())
    return idx_to_class, classes


def expand2square(img, box):
    top, left, bottom, right = box
    w = right - left
    h = bottom - top
    height, width = img.shape[:2]

    if w > h:
        dif = w - h
        d_dif = dif // 2
        top = top - d_dif
        bottom = bottom + d_dif
        pad = 0
        if top < 0:
            pad = -top
            top = 0
            if bottom + pad <= height:
                bottom = bottom + pad
            else:
                bottom = height


        if bottom > height:
            pad = bottom - height
            bottom = height
            if top - pad >= 0:
                top = top - pad
            else:
                top = 0

    elif h > w:
        dif = right - left
        d_dif = dif // 2
        left = left - d_dif
        right = right + d_dif

        if left < 0:
            pad = -left
            left = 0
            if right + pad <= width:
                right = right + pad
            else:
                right = width

        if right > width:
            pad = right - width
            right = width
            if left - pad >= 0:
                left = left - pad
            else:
                left = 0
    new_box = (top, left, bottom, right)
    return new_box


def crop_face(img_path, mode, net):
    img = cv.imread(img_path)
    if mode == "gpu":
        faces, img = fd.detect_face(img, net, mode)
    else:
        faces = fd.detect_face(img, net, mode)

    face_crops = []
    for face in faces:
        face = expand2square(img, face)
        top, left, bottom, right = face
        # print(x, y, w, h)
        img_face = img[int(top):int(bottom), int(left):int(right)]
        face_crops.append(img_face)
    return face_crops


def prepare_model(classes, device, state_dict_path):
    model_ft = models.resnet34(pretrained=True)
    n = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(n, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, len(classes))).to(device)

    model_ft.load_state_dict(state_dict_path)

    model_ft = model_ft.to(device)
    model_ft.eval()
    return model_ft


def detect_race(model, img_name, device, idx_to_class, data_transforms):
    image = Image.open(img_name)
    data_t = data_transforms(image)

    data_t = data_t.to(device)
    outputs_t = model(data_t)
    _, pred_t = torch.max(outputs_t, dim=1)
    label = idx_to_class[pred_t]
    return label


def pipeline():
    PHOTOS_DIR = "some_dir"
    MODEL_PATH = "some_path"
    DONE_DIR = "some_dir2"


    data_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mode = "gpu"
    net = fd.prepare_net(mode)
    device = "cpu"

    idx_to_class, classes = get_classes("models/classes.json")
    model = prepare_model(classes, device, MODEL_PATH)

    for photo in os.listdir(PHOTOS_DIR):
        face_crops = crop_face(os.path.join(PHOTOS_DIR, photo), mode, net)
        done = os.path.join(DONE_DIR, os.path.splitext(photo)[0])
        os.makedirs(done)
        for i, face in enumerate(face_crops):
            cv.imwrite(os.path.join(done, f"{str(i)}.jpg"), face)

        for crop in os.listdir(done):
            crop_path = os.path.join(done, crop)
            label = detect_race(model, crop_path, device, idx_to_class, data_transforms)
            os.rename(crop_path, os.path.join(done, label + "_" + crop))


if __name__ == '__main__':
    pipeline()
