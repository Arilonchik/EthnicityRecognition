from torchvision import models
import torch.nn as nn
from Face_detector import face_detector as fd
import cv2 as cv
from PIL import Image
import torch


class FERModel:
    def __init__(self, config, device):
        self.config = config

        self.state_dict_path = config.weights_path
        self.device = device

        self.data_transforms = config.data_transforms

        self.idx_to_class, self.classes = self.__get_classes()
        self.model = self.__prepare_model()

        self.face_detector = fd.prepare_net("gpu")

    def __prepare_model(self):
        model_ft = models.resnet34(pretrained=True)
        n = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(n, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(self.classes))).to(self.device)

        if torch.cuda.is_available() is False:
            model_ft.load_state_dict(torch.load(self.state_dict_path, map_location=torch.device('cpu')))
        else:
            model_ft.load_state_dict(torch.load(self.state_dict_path))

        model_ft = model_ft.to(self.device)
        model_ft.eval()
        return model_ft

    def __get_classes(self):
        class_to_idx = self.config.classes

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        classes = list(class_to_idx.keys())
        return idx_to_class, classes

    def crop_face(self, img_path, mode):
        img = cv.imread(img_path)
        if mode == "gpu":
            faces, img = fd.detect_face(img, self.face_detector, mode)
        else:
            faces = fd.detect_face(img, self.face_detector, mode)

        face_crops = []
        for face in faces:
            face = self.__expand2square(img, face)
            top, left, bottom, right = face
            img_face = img[int(top):int(bottom), int(left):int(right)]
            face_crops.append(img_face)
        return face_crops

    @staticmethod
    def __expand2square(img, box):
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

    def detect_race(self, img):
        if isinstance(img, str):
            image = Image.open(img)
        else:
            image = img
        data_t = self.data_transforms(image).unsqueeze(0)

        data_t = data_t.to(self.device)
        outputs_t = self.model(data_t)
        _, pred_t = torch.max(outputs_t, dim=1)
        label = self.idx_to_class[pred_t.item()]
        return label

    def process_image(self, image):
        face_crops = self.crop_face(image, "gpu")
        faces = []
        for i, face in enumerate(face_crops):
            img = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            faces.append(im_pil)

        labels = []
        for face in faces:
            label = self.detect_race(face)
            labels.append(label)
        return labels, image
