import torch
import torch.nn.functional as F

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms as transforms

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import urllib
url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'

transform =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = efficientnet_b0(weights = EfficientNet_B0_Weights.IMAGENET1K_V1)
model.to(device)

model.eval()

with urllib.request.urlopen(url) as f:
    labels = [line.decode('utf-8').strip() for line in f.readlines()]

def predict(image, conf=None, iou=None):  # conf/iou unused: classification has no detections/NMS
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # [C, H, W] -> [1, C, H, W ]
    image =  image.to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)
        top5_labels =[labels[i] for i in top5_idx[0].cpu().numpy()]
        top5_scores = top5_prob[0].cpu().numpy().tolist()

    return list(zip(top5_labels, top5_scores))

            
