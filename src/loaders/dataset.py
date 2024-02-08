# import numpy as np
# from torch.utils.data import Dataset
# from PIL import Image
# import json
# import cv2
# from src.utility.utils import get_train_transform, get_valid_transform
# import os
# import torch
# from pycocotools.coco import COCO
 

# class CocoObjectDetection(Dataset):

#     """COCO Custom Dataset compatible with torch.utils.data.DataLoader
#     COCO
#         |-- train
#         |   |-- train.json
#         |   |-- image_id_1.jpg
#         |   |-- image_id_2.jpg
#         |   |-- ...
#         |-- val
#             |-- valid.json
#             |-- image_id_3.jpg
#             |-- image_id_4.jpg
#             |-- ...
#     """

#     def __init__(self,
#                  dataset_folder  : str,
#                  annotation_file : str,
#                  transforms      : dict=None) -> None:

#         """
#         Args
#         :param dataset_folder : Path to Dataset Folder
#         :param annotation_file: Path to Annotation File
#         :param transform      : Transformation Pipeline from Compose
#         """

#         self.dataset_folder  = dataset_folder
#         self.annotation_file = annotation_file
#         self.coco =  COCO(annotation_file=self.annotation_file)
#         self.check_ann()
#         self.ids  =  list(self.coco.getImgIds())
#         self.transforms = transforms
#         self.info()

#     def info(self):
#         print(f"Number of images in dataset : {len(self.coco.getImgIds())}")
#         print(f"Number of Categories: {len(self.coco.loadCats(self.coco.getCatIds()))}")

#     def check_ann(self) -> None:
#         annotated_image_ids = set(annotation['image_id']
#                 for annotation in self.coco.dataset['annotations'])
#         all_image_ids = set(image['id']
#                             for image in self.coco.dataset['images'])
#         unannotated_image_ids = all_image_ids - annotated_image_ids
#         file_names = [self.coco.dataset["images"][i]["file_name"]
#                       for i in unannotated_image_ids]
#         if len(file_names)>0:
#             print("Unannotated Images : ",*file_names)
#             print("Skipping annotations ...")
#             filtered_annotations = [annotation for annotation in self.coco.dataset['annotations'] if annotation['image_id'] not in unannotated_image_ids]
#             self.coco.dataset["annotation"] = filtered_annotations

#     def __getitem__(self, item:int) -> tuple:

#         """Returns one data pair (image and annotations)."""

#         image_id = self.ids[item]
#         annotation_id = self.coco.getAnnIds(imgIds=image_id)
#         annotation = self.coco.loadAnns(ids=annotation_id)
#         path = self.coco.loadImgs(image_id)[0]["file_name"]
#         img = cv2.imread(os.path.join(self.dataset_folder, path))
#         num_of_instance = len(annotation)

#         boxes    = []
#         labels   = []
#         area     = []
#         is_crowd = []

#         for index,instance in enumerate(annotation):
#             x = instance['bbox'][0]
#             y = instance['bbox'][1]
#             w = x + instance['bbox'][2]
#             h = x + instance['bbox'][3]
#             boxes.append([x, y, w, h])
#             area.append(instance['area'])
#             is_crowd.append(instance['iscrowd'])
#             labels.append(instance['category_id'])

#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int8)
#         area = torch.as_tensor(area, dtype=torch.float32)
#         is_crowd = torch.as_tensor(is_crowd, dtype=torch.int64)

#         my_annotation = {}
#         my_annotation["boxes"] = boxes
#         my_annotation["labels"] = labels
#         my_annotation["image_id"] = torch.tensor([image_id])
#         my_annotation["area"] = area
#         my_annotation["is_crowd"] = is_crowd

#         if self.transforms is not None:
#             img = self.transforms(image=img)["image"]
            

#         return img, my_annotation

#     def __len__(self):
#         return len(self.coco.getImgIds())

# train_dataset = CocoObjectDetection(dataset_folder="data/raw/PPE/train", annotation_file="data/raw/PPE/train/_annotations.coco.json",transforms=get_train_transform())

import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from src.utility.utils import collate_fn, get_train_transform, get_valid_transform
from config.config import settings

CLASSES = settings["train"].CLASSES
BATCH_SIZE = settings["train"].BATCH_SIZE
TRAIN_DIR = settings["train"].TRAIN_DIR
VALID_DIR = settings["train"].VALID_DIR
RESIZE_TO = settings["train"].IMG_SIZE


# the dataset class
class MicrocontrollerDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        
        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# prepare the final datasets and data loaders
train_dataset = MicrocontrollerDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = MicrocontrollerDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")
