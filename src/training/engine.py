import torch
import torch.nn as nn

from collections import defaultdict
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from typing import Dict, List, Union, Tuple, Any

def create_resnet18_model():
    """
    Returns a ResNet-18 model with the final layer adapted to CIFAR-10 classes.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10) # Adapt final layer to CIFAR-10 classes
    return model

def calculate_metrics(
                    true_positive: Dict[str, int],
                    false_positive: Dict[str, int],
                    false_negative: Dict[str, int],
                    total:Dict[str, int],
                    classes: List[str]
                    ):
    """
    Calculate the precision, recall, f1 score, and accuracy for each class.

    Args:
        true_positive (Dict[str, int]): The number of true positives for each class.
        false_positive (Dict[str, int]): The number of false positives for each class.
        false_negative (Dict[str, int]): The number of false negatives for each class.
        total (Dict[str, int]): The total number of examples for each class.
        classes (List[str]): The list of class names.
    """
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
                    model:torch.nn.Module,
                    criterion:torch.nn.Module,
                    optimiser:torch.optim.Optimizer,
                    data_loader:torch.utils.data.DataLoader,
                    epoch:int,
                    num_batches:int,
                    classes:List[str],
                    device:Union[str, torch.device],
                    print_interval:int,
                    mode:str="train"
                    ) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Helper function to perform a forward pass for a single epoch.

    Args:
        model (torch.nn.Module): The model to train or evaluate.
        criterion (torch.nn.Module): The loss function to use.
        optimiser (torch.optim.Optimizer): The optimiser to use.
        data_loader (torch.utils.data.DataLoader): The data loader to use.
        epoch (int): The current epoch number.
        num_batches (int): The number of batches in the data loader.
        classes (List[str]): The list of class names.
        device (Union[str, torch.device]): The device to use.
        print_interval (int): The interval to print the loss.
        mode (str): The mode to use. Either "train" or "val".
    """

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
    
def train_model(
                model:torch.nn.Module,
                criterion:torch.nn.Module,
                optimiser:torch.optim.Optimizer,
                scheduler:torch.optim.lr_scheduler._LRScheduler,
                train_dl:torch.utils.data.DataLoader,
                val_dl:torch.utils.data.DataLoader,
                num_epochs:int,
                classes:List[str],
                device:Union[str, torch.device],
                print_interval:int
                ) -> torch.nn.Module:
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_precision, train_recall, train_f1_score, train_accuracy = epoch_forward_pass(
                                                                                        model=model,
                                                                                        criterion=criterion,
                                                                                        optimiser=optimiser,
                                                                                        data_loader=train_dl,
                                                                                        epoch=epoch,
                                                                                        num_batches=len(train_dl),
                                                                                        classes=classes,
                                                                                        device=device,
                                                                                        print_interval=print_interval,
                                                                                        mode="train"
                                                                                        )
        # Print metrics for each class:
        print(f"Epoch: {epoch + 1}")
        print(f"Training metrics")
        print(f"Average train loss: {train_loss:.4f}")
        for class_name in classes:
            print(f"Class: {class_name} | Precision: {train_precision[class_name]:.2f} | Recall: {train_recall[class_name]:.2f} | F1 Score: {train_f1_score[class_name]:.2f} | Accuracy: {train_accuracy[class_name]:.2f}")
        
        # Validate model after each epoch
        model.eval()
        with torch.no_grad():
            val_loss, val_precision, val_recall, val_f1_score, val_accuracy = epoch_forward_pass(
                                                                                                model=model,
                                                                                                criterion=criterion,
                                                                                                optimiser=optimiser,
                                                                                                data_loader=val_dl,
                                                                                                epoch=epoch,
                                                                                                num_batches=len(val_dl),
                                                                                                classes=classes,
                                                                                                device=device,
                                                                                                print_interval=print_interval,
                                                                                                mode="val"
                                                                                                )
        # Print metrics for each class:
        print(f"Validation metrics")
        print(f"Average val loss: {val_loss:.4f}")
        for class_name in classes:
            print(f"Class: {class_name} | Precision: {val_precision[class_name]:.2f} | Recall: {val_recall[class_name]:.2f} | F1 Score: {val_f1_score[class_name]:.2f} | Accuracy: {val_accuracy[class_name]:.2f}")
        print("\n")

        # Update scheduler after each epoch
        scheduler.step()

    return model

def test_model(
                model:torch.nn.Module,
                test_dl:torch.utils.data.DataLoader,
                classes:List[str],
                device:Union[str, torch.device]
                ):

    total = {class_name: 0 for class_name in classes}
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    model.eval()

    with torch.no_grad():
        for data in test_dl:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            _, predicted = torch.max(output, 1)
            
            for label, prediction in zip(labels, predicted):
                total[classes[label]] += 1 # Track total for accuracy

                if label == prediction:
                    true_positive[classes[label]] += 1
                else:
                    false_positive[classes[prediction]] += 1 # Predicted class is wrong
                    false_negative[classes[label]] += 1 # Actual class is wrong

    return true_positive, false_positive, false_negative, total