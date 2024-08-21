# go through dataset, try to load all images, faulty images: print path

import glob
from PIL import Image
import PIL
from tqdm import tqdm


# save list of faulty images to file
def save(list_of_wrong_images):
    try:
        with open("data/GenImage/faulty_images.txt", "w") as f:
            for image in list_of_wrong_images:
                f.write(image + "\n")
    except:
        print(list_of_wrong_images)


list_of_wrong_images = []
# all png, jpeg, jpg images
for file in tqdm(glob.iglob("data/GenImage/*/*/*/*/*.png")):
    try:
        Image.open(file)
    except PIL.UnidentifiedImageError:
        list_of_wrong_images.append(file)
        save(list_of_wrong_images)

for file in tqdm(glob.iglob("data/GenImage/*/*/*/*/*.jpeg")):
    try:
        Image.open(file)
    except PIL.UnidentifiedImageError:
        list_of_wrong_images.append(file)
        save(list_of_wrong_images)

for file in tqdm(glob.iglob("data/GenImage/*/*/*/*/*.jpg")):
    try:
        Image.open(file)
    except PIL.UnidentifiedImageError:
        list_of_wrong_images.append(file)
        save(list_of_wrong_images)
