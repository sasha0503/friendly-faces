from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)  # keep all other parameters default


def detect(image_path):
    rgba_image = Image.open(image_path)
    rgb_image = rgba_image.convert("RGB")

    img_cropped = mtcnn(rgb_image, save_path=f'data/crops/{os.path.basename(image_path)}.png')


def save_crops(data_folder):
    data_path = data_folder
    for person in os.listdir(data_path):
        print(f'Processing {person}')
        person_path = os.path.join(data_path, person)
        if not os.path.isdir(person_path):
            continue
        for image in os.listdir(person_path):
            image_path = os.path.join(person_path, image)
            try:
                detect(image_path)
            except:
                print(f'Error processing: {image_path}')
                continue


def cool_grid(data_folder):
    image_files = [file for file in os.listdir(data_folder)]
    images = []

    for img_file in image_files:
        img_path = os.path.join(data_folder, img_file)
        img = Image.open(img_path)
        images.append(img)

    target_size = (200, 200)  # Adjust the size as needed

    resized_images = []
    for img in images:
        resized_img = img.resize(target_size)
        resized_images.append(resized_img)

    rows = 9
    columns = 10
    grid_width = target_size[0] * columns
    grid_height = target_size[1] * rows

    grid_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

    for row in range(rows):
        for col in range(columns):
            index = row * columns + col
            if index < len(resized_images):
                x_offset = col * target_size[0]
                y_offset = row * target_size[1]
                grid_image.paste(resized_images[index], (x_offset, y_offset))

    grid_image.save('data/grid.png')


if __name__ == '__main__':
    # save_crops('data')
    cool_grid('data/crops')
