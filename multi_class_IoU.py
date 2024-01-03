# Pseudo code to calculate IoU for multi-class segmentation
from tensorflow.keras import backend as K
import numpy as np

def calculate_intersection(y_true_class, y_pred_class):
    # Perform element-wise logical "AND" operation to get the intersection
    intersection = np.logical_and(y_true_class, y_pred_class)
    
    # Count the number of True values (pixels) in the intersection mask
    intersection_count = np.sum(intersection)
    
    return intersection_count


def calculate_union(y_true_class, y_pred_class, intersection):
    # Count the number of True values (pixels) in the ground truth and predicted masks
    y_true_count = np.sum(y_true_class)
    y_pred_count = np.sum(y_pred_class)
    
    # Calculate the union using the sizes of the masks and the precomputed intersection
    union = y_true_count + y_pred_count - intersection
    
    return union

def calculate_iou_for_class(y_true, y_pred, class_label, epsilon=1e-6):
    # Get binary masks for the specified class
    y_true_class = (y_true == class_label)
    y_pred_class = (y_pred == class_label)
    
    # Calculate IoU for the specified class using the formula you have (e.g., jacard_coef)
    intersection = calculate_intersection(y_true_class, y_pred_class)
    union = calculate_union(y_true_class, y_pred_class, intersection)
    iou_class = intersection / (union + epsilon)  # Add epsilon for smoothing
    
    return iou_class


def calculate_mean_iou_for_classes(y_true, y_pred, class_labels):
    # Initialize variables to store IoU values and counts for relevant classes
    iou_scores = []
    for class_label in class_labels: 
        # Calculate IoU for the current class label
        iou_class = calculate_iou_for_class(y_true, y_pred, class_label)
        print(class_label, iou_class)
        iou_scores.append(iou_class)
    
    return iou_scores
        # Accumulate IoU and count for the relevant classes
        # iou_sum += iou_class
        # class_count += 1   
    # Calculate mean IoU across relevant classes (excluding 'normal' class)
    # mean_iou = iou_sum / class_count if class_count > 0 else 0.0
    # return mean_iou

#100_eva_3c_test_fold_2.txt
def iou_for_classes(y_true, y_pred, class_labels, smooth=100):
    iou_scores = []
    for class_label in class_labels:
        y_true_class = K.cast(y_true == class_label, dtype=K.floatx())
        y_pred_class = K.cast(y_pred == class_label, dtype=K.floatx())
        
        intersection = K.sum(K.abs(y_true_class * y_pred_class), axis=-1)
        sum_ = K.sum(K.square(K.cast(y_true_class, 'float32')), axis=-1) + \
            K.sum(K.square(K.cast(y_pred_class, 'float32')), axis=-1)

        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        print(class_label, jac)

        iou_scores.append(jac)
    return iou_scores


def calculate_dice_for_classes(y_true, y_pred, class_labels, epsilon=1e-6):
    dice_scores = []
    
    # Iterate through specified class labels (excluding 'normal' class)
    for class_label in class_labels:
        y_true_class = (y_true == class_label)
        y_pred_class = (y_pred == class_label)
        
        intersection = np.sum(y_true_class & y_pred_class)
        dice = (2.0 * intersection) / (np.sum(y_true_class) + np.sum(y_pred_class) + epsilon)
        print(class_label, dice)
        
        dice_scores.append(dice)
    
    # Calculate mean Dice across specified classes
    # mean_dice = np.mean(dice_scores)
    
    return dice_scores

def precision(y_true, y_pred, class_labels):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    precision_scores = []
    for class_label in class_labels:
        y_true_class = K.cast(y_true == class_label, dtype=K.floatx())
        y_pred_class = K.cast(y_pred == class_label, dtype=K.floatx())

        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        print(class_label, precision)

        precision_scores.append(precision)

    return precision_scores


def recall(y_true, y_pred, class_labels):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    recall_scores = []
    for class_label in class_labels:
        y_true_class = K.cast(y_true == class_label, dtype=K.floatx())
        y_pred_class = K.cast(y_pred == class_label, dtype=K.floatx())

        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        print(class_label, precision)

        recall_scores.append(recall)
    return recall_scores

def accuracy(y_true, y_pred, class_labels):
    accuracy_scores = []
    for class_label in class_labels:
        y_true_class = K.cast(y_true == class_label, dtype=K.floatx())
        y_pred_class = K.cast(y_pred == class_label, dtype=K.floatx())
        
        accuracy = K.mean(K.equal(y_true_class, K.round(y_pred_class)))
        print(class_label, accuracy)

        accuracy_scores.append(accuracy)
    return accuracy_scores


# Assuming 'y_true' and 'y_pred' are your ground truth and predicted masks
# 'class_labels' specifies the classes of interest (excluding 'normal')
# class_labels = [1, 2]  # Gleason patterns 3 and 4

# Usage example:
# iou_score = calculate_mean_iou(y_true_masks, y_pred_masks, class_labels)  
# print(f"Mean IoU excluding 'normal' class: {iou_score}")

# Calculate Dice coefficient excluding 'normal' class
# dice_score = calculate_dice_for_classes(y_true, y_pred, class_labels)
# print(f"Dice coefficient excluding 'normal' class: {dice_score}")




