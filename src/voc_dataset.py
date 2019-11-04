"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from src.data_augmentation import *


class VOCDataset(Dataset):
    def __init__(self, root_path="data/VOCdevkit", year="2007", mode="train", image_size=448, is_training = True):
        if (mode in ["train", "val", "trainval", "test"] and year == "2007") or (
                mode in ["train", "val", "trainval"] and year == "2012"):
            self.data_path = os.path.join(root_path, "VOC{}".format(year))
            id_list_path = os.path.join(self.data_path, "ImageSets/Main/{}.txt".format(mode))
            self.image_paths = [os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id.strip())) for id in open(id_list_path)]
            self.xml_paths = [os.path.join(self.data_path, "Annotations", "{}.xml".format(id.strip())) for id in open(id_list_path)]
        elif mode == "test" and year == "0712":
            self.data_path = os.path.join(root_path, "VOC2007")
            id_list_path = os.path.join(self.data_path, "ImageSets/Main/{}.txt".format(mode))
            self.image_paths = [os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id.strip())) for id in
                                open(id_list_path)]
            self.xml_paths = [os.path.join(self.data_path, "Annotations", "{}.xml".format(id.strip())) for id in
                              open(id_list_path)]
        elif mode == "trainval" and year == "0712":
            id_list_path1 = os.path.join(root_path, 'VOC2007/ImageSets/Main/trainval.txt')
            id_list_path2 = os.path.join(root_path, 'VOC2012/ImageSets/Main/trainval.txt')
            self.image_paths = [os.path.join(root_path, "VOC2007/JPEGImages", "{}.jpg".format(id.strip())) for id in
                                open(id_list_path1)] + [os.path.join(root_path, "VOC2012/JPEGImages", "{}.jpg".format(id.strip())) for id in
                                open(id_list_path2)]
            self.xml_paths = [os.path.join(root_path, "VOC2007/Annotations", "{}.xml".format(id.strip())) for id in
                              open(id_list_path1)] + [os.path.join(root_path, "VOC2012/Annotations", "{}.xml".format(id.strip())) for id in
                                open(id_list_path2)]
        else:
            raise Exception('Wrong year number or mode for VOC dataset!!!')

        # id_list_path = os.path.join(self.data_path, "ImageSets/Main/{}.txt".format(mode))
        # self.ids = [id.strip() for id in open(id_list_path)]
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.image_paths)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image_xml_path = self.xml_paths[item]
        # image_path = os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_xml_path = os.path.join(self.data_path, "Annotations", "{}.xml".format(id))
        annot = ET.parse(image_xml_path)

        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])
        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
