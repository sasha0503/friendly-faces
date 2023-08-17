import os

import torch

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)  # keep all other parameters default
resnet = InceptionResnetV1(pretrained='vggface2').eval()

warmup_image = Image.open('data/warmup.png').convert("RGB")
warmup_image_cropped = mtcnn(warmup_image)
warmup_image_embedding = resnet(warmup_image_cropped.unsqueeze(0))


def detect(image_path):
    rgba_image = Image.open(image_path)
    rgb_image = rgba_image.convert("RGB")

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(rgb_image)

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))

    # Or, if using for VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))

    return img_probs, img_embedding


if __name__ == '__main__':
    images = os.listdir('data')
    res = {}
    for image in images:
        img_probs, img_embedding = detect('data/' + image)
        res[image] = (img_probs, img_embedding)

    res = {}
    for image in images:
        img_probs, img_embedding = detect('data/' + image)
        res[image] = (img_probs, img_embedding)

    ground_truth = res['lr_1.png']
    slavko = res['slavko_1.png']
    lr_2 = res['lr_3.png']

    print(f'Slavko emb: {torch.dist(ground_truth[1], slavko[1])}')
    print(f'LR_2 emb: {torch.dist(ground_truth[1], lr_2[1])}')

    print(f'Slavko prob: {torch.dist(ground_truth[0], slavko[0])}')
    print(f'LR_2 prob: {torch.dist(ground_truth[0], lr_2[0])}')
