import torch
import torch.nn as nn

from collections import defaultdict
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

def create_resnet18_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10) # Adapt final layer to CIFAR-10 classes
    return model

def calculate_metrics(
                    true_positive, 
                    false_positive, 
                    false_negative, 
                    total, 
                    classes
                    ):
    precision = {}
    recall = {}
    f1_score = {}
    accuracy = {}
        
    for class_name in classes:
        tp = true_positive[class_name]
        fp = false_positive[class_name]
        fn = false_negative[class_name]
        total_count = total[class_name]

        # Accuracy (same as existing code)
        accuracy[class_name] = 100 * float(tp) / total_count if total_count != 0 else 0

        # Precision = TP / (TP + FP)
        precision[class_name] = tp / (tp + fp) if (tp + fp) != 0 else 0

        # Recall = TP / (TP + FN)
        recall[class_name] = tp / (tp + fn) if (tp + fn) != 0 else 0

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        if (precision[class_name] + recall[class_name]) > 0:
            f1_score[class_name] = 2 * (precision[class_name] * recall[class_name]) / (precision[class_name] + recall[class_name])
        else:
            f1_score[class_name] = 0

    return precision, recall, f1_score, accuracy

def epoch_forward_pass(
                    model, 
                    criterion, 
                    optimiser, 
                    data_loader, 
                    epoch, 
                    num_batches, 
                    classes, 
                    device, 
                    print_interval, 
                    mode="train"
                    ):

    total = {class_name: 0 for class_name in classes}
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    running_loss = 0.0
    total_examples = 0

    for i, data in enumerate(data_loader):
        inputs, labels = data
        optimiser.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)

        # Count true positives, false positives, and false negatives
        _, predicted = torch.max(output, 1)
        
        for label, prediction in zip(labels, predicted):
            total[classes[label]] += 1 # Track total for accuracy

            if label == prediction:
                true_positive[classes[label]] += 1
            else:
                false_positive[classes[prediction]] += 1 # Predicted class is wrong
                false_negative[classes[label]] += 1 # Actual class is wrong
        
        loss = criterion(output, labels)
        
        if mode == "train":
            loss.backward()
            optimiser.step()

        running_loss += loss.item()
        total_examples += len(labels)

        if (i % print_interval == (print_interval - 1)) or (i == num_batches - 1):
            current_lr = optimiser.param_groups[0]["lr"]
            print(f"Epoch: {epoch + 1} |  Batch: {i + 1}/{num_batches} | Loss: {running_loss / total_examples} | LR: {current_lr}")

    # Calculate metrics after each epoch
    precision, recall, f1_score, accuracy = calculate_metrics(
                                                            true_positive=true_positive, 
                                                            false_positive=false_positive, 
                                                            false_negative=false_negative, 
                                                            total=total,
                                                            classes=classes
                                                            )
    average_epoch_loss = running_loss / total_examples
    return average_epoch_loss, precision, recall, f1_score, accuracy
    
