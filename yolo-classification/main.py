import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, num_boxes=2):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(192, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 * 7 * (num_classes + num_boxes * 5)),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def yolo_loss(predictions, target, lambda_coord=5, lambda_noobj=0.5):
    """
    Compute the YOLO loss.

    Args:
        predictions (Tensor): The output tensor from the YOLO network.
        target (Tensor): The ground truth tensor.
        lambda_coord (float): The weight for the localization loss.
        lambda_noobj (float): The weight for the no-object loss.

    Returns:
        Tensor: The YOLO loss.
    """
    batch_size = predictions.size(0)
    num_boxes = predictions.size(2)
    num_classes = predictions.size(3) // 5

    # Reshape the predictions and target tensors
    predictions = predictions.view(batch_size, num_boxes, num_classes + 5)
    target = target.view(batch_size, num_boxes, num_classes + 5)

    # Compute the classification loss
    class_pred = predictions[:, :, :num_classes]
    class_target = target[:, :, :num_classes]
    class_mask = torch.ByteTensor(batch_size, num_boxes).fill_(0)
    class_loss = F.mse_loss(class_pred[class_target != 0], class_target[class_target != 0], reduction='sum')
    # Compute the localization loss
    box_pred = predictions[:, :, num_classes:]
    box_target = target[:, :, num_classes:]
    obj_mask = torch.ByteTensor(batch_size, num_boxes).fill_(0)
    obj_mask[class_target != 0] = 1
    loc_loss = lambda_coord * F.mse_loss(box_pred[obj_mask], box_target[obj_mask], reduction='sum')

    # Compute the confidence loss
    conf_pred = predictions[:, :, num_classes:num_classes + 1]
    conf_target = target[:, :, num_classes:num_classes + 1]
    noobj_mask = torch.ByteTensor(batch_size, num_boxes).fill_(1)
    noobj_mask[obj_mask] = 0
    noobj_loss = lambda_noobj * F.mse_loss(conf_pred[noobj_mask], conf_target[noobj_mask], reduction='sum')

    # Compute the total loss
    total_loss = class_loss + loc_loss + noobj_loss

    return total_loss

def yolo_detect(predictions, conf_thresh=0.5, nms_thresh=0.4):
    """
    Perform object detection using the YOLO predictions.

    Args:
        predictions (Tensor): The output tensor from the YOLO network.
        conf_thresh (float): The confidence threshold for filtering bounding boxes.
        nms_thresh (float): The threshold for non-maximum suppression.

    Returns:
        List: A list of bounding boxes and class probabilities.
    """
    batch_size = predictions.size(0)
    num_boxes = predictions.size(2)
    num_classes = predictions.size(3) // 5

    # Reshape the predictions tensor
    predictions = predictions.view(batch_size, num_boxes, num_classes + 5)

    # Extract the class probabilities and bounding box coordinates
    class_probs = predictions[:, :, :num_classes]
    box_coords = predictions[:, :, num_classes:]

    # Apply confidence thresholding
    conf_mask = (class_probs.max(2)[0] > conf_thresh).float().unsqueeze(2)
    class_probs = class_probs * conf_mask

    # Apply non-maximum suppression
    detections = []
    for i in range(batch_size):
        boxes_i = box_coords[i]
        scores_i = class_probs[i]
        boxes, classes, scores = [], [], []

        for c in range(num_classes):
            class_boxes = boxes_i[scores_i[:, c] > conf_thresh]
            class_boxes = class_boxes[scores_i[:, c] > conf_thresh]
            class_scores = scores_i[:, c][scores_i[:, c] > conf_thresh]

            if class_boxes.size(0) == 0:
                continue

            _, sorted_indices = class_scores.sort(descending=True)
            class_boxes = class_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]

            keep = torch.tensor([], dtype=torch.long)
            for j in range(class_boxes.size(0)):
                if nms_thresh == 0:
                    keep = torch.cat((keep, torch.tensor([j])))
                    continue

                box = class_boxes[j].unsqueeze(0)
                iou = calc_iou(box, class_boxes)
                keep = torch.cat((keep, torch.tensor([j] + (iou < nms_thresh).nonzero().squeeze().tolist())))

            boxes.append(class_boxes[keep])
            classes.append(torch.full_like(keep, c))
            scores.append(class_scores[keep])

        if boxes:
            boxes = torch.cat(boxes, dim=0)
            classes = torch.cat(classes)
            scores = torch.cat(scores)
            detections.append((boxes, classes, scores))
        else:
            detections.append(None)

    return detections

def calc_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (Tensor): Bounding box 1 in [x1, y1, x2, y2] format.
        box2 (Tensor): Bounding box 2 in [x1, y1, x2, y2] format.

    Returns:
        Tensor: The IoU value between the two bounding boxes.
    """
    x1, y1, x2, y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x1_b, y1_b, x2_b, y2_b = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    x1 = torch.max(x1, x1_b)
    y1 = torch.max(y1, y1_b)
    x2 = torch.min(x2, x2_b)
    y2 = torch.min(y2, y2_b)

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = box1_area + box2_area - intersection
    iou = intersection / union

    return iou


