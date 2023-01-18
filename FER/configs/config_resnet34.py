from torchvision import transforms

version = "0.2.2"

classes = {
    "East Asian": 0,
    "Indian": 1,
    "Black": 2,
    "White": 3,
    "Middle Eastern": 4,
    "Latino_Hispanic": 5,
    "Southeast Asian": 6
  }

model_type = "resnet34"
weights_path = "FER/weights/model_dict_resnet34.pt"

data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
