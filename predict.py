import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from src.model.resnet import SiameseNetwork
from src.feature_extractor.extractor import FeatureExtractor
from config.config import settings
from loaders.dataset import build_transformer
import cv2
import time


class Inference:

    """
    Inference class implementation

    """
    def __init__(self, dataset:str, embedding_path:str, model, transform=None):
        self.__model = model
        self.__extractor = FeatureExtractor(model=self.__model, data_dir=dataset,transform=transform)
        self.__embeddings_dict = self.__extractor.extraction(feature_dir=embedding_path)
        self.__transform = transform


    def __load_and_preprocess_image(self, image:np.ndarray)->bool:
        """
        Preprocess step before inference
        """
        image = self.__transform(Image.fromarray(image))
        return image
    

    def __predict(self,query_image:np.ndarray) -> str:

        """_summary_

        Args:
            query_image (np.ndarray): _description_

        Returns:
            str: _description_
        """
        
        gray_frame = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        if self.__is_object_present(gray_frame):
            similarity_score = []
            query_image = self.__load_and_preprocess_image(gray_frame)
            with torch.no_grad():
                query_embedding = self.__model.forward_once(query_image.cuda().unsqueeze(0))
            for means in self.__embeddings_dict:
                euclidean_distance = F.pairwise_distance(query_embedding, self.__embeddings_dict[means])
                similarity_score.append(euclidean_distance)
            if similarity_score.index(min(similarity_score))+1 == 1:
                        return "front"
            else:
                 return "back"
        else:
            return "empty"
        
    def __is_object_present(self, roi_image:np.ndarray , threshold_value:int=40)-> bool:
        _, thresh = cv2.threshold(roi_image,80,255,cv2.THRESH_BINARY_INV)
        avg_color_per_row = np.average(thresh, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        print(avg_color)
        if avg_color >= threshold_value:
            return False
        else:
            return True
    
    def get_prediction(self,query_image:np.ndarray, image_config:dict)-> np.ndarray:

        """ 
        Args:
            query_image (np.ndarray): 
            image_config (dict): 

        Returns:
            np.ndarray: 
        """

        for b_no, value in image_config.items():
            result = self.__predict(query_image=query_image[value["y"]:
                                                            value["y"]+value["h"],
                                                            value["x"]:
                                                            value["x"]+value["w"]])
            query_image = cv2.rectangle(query_image, (value["x"], 
                                                      value["y"]),
                                                     (value["x"]+value["w"], 
                                                      value["y"]+value["h"]),
                                                      color=(0,200,95),
                                                      thickness=2,
                                                      lineType=cv2.LINE_4)
            query_image = cv2.putText(query_image, result, 
                                      (value["x"], value["y"] - 10),
                                      cv2.FONT_HERSHEY_DUPLEX,
                                      0.5,
                                      (10,200,140),
                                      1,
                                      cv2.LINE_8)
        return query_image


# Model Creation
device = "cuda" if torch.cuda.is_available() else "cpu"
net = SiameseNetwork().to(device)
train_state_dict = torch.load(f="restore/checkpoints/resnet-saar.pth")
net.load_state_dict(train_state_dict)
        
infer_engine = Inference(model=net,
                         dataset=settings["feature"]["feature_set"],
                         embedding_path=settings["feature"]["feature_dir"],
                         transform=build_transformer(image_size=100)["valid"])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -4.0)
while True:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, 1)
    frame = infer_engine.get_prediction(frame, settings["rois"])
    time.sleep(0.1)
    cv2.imshow("result",frame)
    cv2.imwrite("Frame.jpg", frame)
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()