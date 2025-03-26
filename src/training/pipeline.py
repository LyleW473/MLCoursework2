import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import pickle

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, List, Union, Tuple

from src.utils import load_active_learning_embeddings
from src.training.engine import create_resnet18_model, train_model, test_model, calculate_metrics
from src.training.dataset import CustomDataset, LinearEvalDataset

class FullySupervisedTrainingPipeline:
    def __init__(
                self, 
                training_settings:Dict[str, Any],
                classes:List[str],
                transform:torchvision.transforms.Compose, 
                device:Union[str, torch.device]
                ):
        """
        Initialises the TrainingPipeline object with the training settings, classes, transform and device.

        Args:
            training_settings (Dict[str, Any]): The training settings for the model.
            classes (List[str]): The list of classes in the dataset.
            transform (torchvision.transforms.Compose): The transform to apply to the images.
            device (Union[str, torch.device]): The device to run the training on.
        """
        self.training_settings = training_settings
        self.classes = classes
        self.transform = transform
        self.device = device
    
    def load_data(self, embeddings_dir:str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

        all_images, all_labels, _ = load_active_learning_embeddings(embeddings_dir)

        # Separate into train and validation sets:
        train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=2004)

        print(f"Number of training images: {len(train_images)}")
        print(f"Number of validation images: {len(val_images)}")

        # train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_set = CustomDataset(images=train_images, labels=train_labels, transform=self.transform)
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=self.training_settings["batch_size"], shuffle=True, num_workers=0)

        val_set = CustomDataset(images=val_images, labels=val_labels, transform=self.transform)
        val_dl = torch.utils.data.DataLoader(val_set, batch_size=self.training_settings["batch_size"], shuffle=False, num_workers=0)

        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=self.transform)
        test_dl = torch.utils.data.DataLoader(test_set, batch_size=self.training_settings["batch_size"], shuffle=False, num_workers=2)

        return train_dl, val_dl, test_dl
    
    def initialise_components(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Initialises the model, criterion, optimiser and scheduler.
        """
        model = create_resnet18_model()
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimiser = optim.SGD(model.parameters(), lr=self.training_settings["lr"], momentum=self.training_settings["momentum"], nesterov=self.training_settings["use_nesterov"])
        scheduler = CosineAnnealingLR(optimiser, T_max=self.training_settings["n_epochs"]) # T_max is the number of epochs

        return model, criterion, optimiser, scheduler

    def execute(self, version:str, setting:str, embeddings_dir:str) -> None:
        """
        Executes a single training run for the given version and setting, using the embeddings in the given directory.

        Args:
            version (str): The version of the model.
            setting (str): The setting of the model.
            embeddings_dir (str): The directory containing the embeddings of the images.
        
        """
        train_dl, val_dl, test_dl = self.load_data(embeddings_dir=embeddings_dir)

        model, criterion, optimiser, scheduler = self.initialise_components()

        model = train_model(
                            model=model,
                            criterion=criterion,
                            optimiser=optimiser,
                            scheduler=scheduler,
                            train_dl=train_dl,
                            val_dl=val_dl,
                            num_epochs=self.training_settings["n_epochs"],
                            classes=self.classes,
                            device=self.device,
                            print_interval=100
                            )
    
        # Save model
        split_dir = embeddings_dir.split("/")
        iterations_and_b_setting = split_dir[-1]
        MODEL_SAVE_DIR = "models"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_path = f"{MODEL_SAVE_DIR}/resnet18_{version}_{setting}_{iterations_and_b_setting}.pth"
        torch.save(model.state_dict(), model_path)

        # Test model
        saved_model = create_resnet18_model()
        saved_model.load_state_dict(torch.load(model_path))
        saved_model.to(self.device)

        true_positive, false_positive, false_negative, total = test_model(
                                                                        model=saved_model,
                                                                        test_dl=test_dl,
                                                                        classes=self.classes,
                                                                        device=self.device
                                                                        )
        precision, recall, f1_score, accuracy = calculate_metrics(
                                                            true_positive=true_positive, 
                                                            false_positive=false_positive, 
                                                            false_negative=false_negative, 
                                                            total=total,
                                                            classes=self.classes,
                                                            )
        total_accuracy = sum(true_positive.values()) / sum(total.values())

        completed_dict = {
                        "model": saved_model.state_dict(),
                        "metrics": {
                                    "precision": precision,
                                    "recall": recall,
                                    "f1_score": f1_score,
                                    "accuracy": accuracy,
                                    "total_accuracy": total_accuracy
                                    }
                        }
        print("Completed training for version:", version, "and setting:", setting)

        # Save the completed dict in the same location (overwrite)
        with open(f"{MODEL_SAVE_DIR}/completed_{version}_{setting}_{iterations_and_b_setting}.pth", "wb") as f:
            torch.save(completed_dict, f)

class LinearEvalPipeline:
    def __init__(
                self, 
                training_settings:Dict[str, Any],
                classes:List[str],
                transform:torchvision.transforms.Compose, 
                device:Union[str, torch.device]
                ):
        """
        Initialises the TrainingPipeline object with the training settings, classes, transform and device.

        Args:
            training_settings (Dict[str, Any]): The training settings for the model.
            classes (List[str]): The list of classes in the dataset.
            transform (torchvision.transforms.Compose): The transform to apply to the images.
            device (Union[str, torch.device]): The device to run the training on.
        """
        self.training_settings = training_settings
        self.classes = classes
        self.transform = transform
        self.device = device

    def load_data(self, embeddings_dir:str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

        _, all_labels, all_embeddings = load_active_learning_embeddings(embeddings_dir)

        # Separate into train and validation sets:
        train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(all_embeddings, all_labels, test_size=0.2, random_state=2004)

        print(f"Number of training images: {len(train_embeddings)}")
        print(f"Number of validation images: {len(val_embeddings)}")
        
        train_set = LinearEvalDataset(embeddings=train_embeddings, labels=train_labels, transform=self.transform)
        train_dl = torch.utils.data.DataLoader(train_set, batch_size=self.training_settings["batch_size"], shuffle=True, num_workers=0)

        val_set = LinearEvalDataset(embeddings=val_embeddings, labels=val_labels, transform=self.transform)
        val_dl = torch.utils.data.DataLoader(val_set, batch_size=self.training_settings["batch_size"], shuffle=False, num_workers=0)


        with open("embeddings/simclr_cifar10_test_embeddings.pkl", "rb") as f:
            test_embeddings_dict = pickle.load(f)
        test_embeddings = [test_embeddings_dict[i]["embedding"] for i in range(len(test_embeddings_dict))]
        test_labels = [test_embeddings_dict[i]["label"] for i in range(len(test_embeddings_dict))]
        test_set = LinearEvalDataset(embeddings=test_embeddings, labels=test_labels, transform=self.transform)
        test_dl = torch.utils.data.DataLoader(test_set, batch_size=self.training_settings["batch_size"], shuffle=False, num_workers=2)
        return train_dl, val_dl, test_dl

    def initialise_components(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Initialises the model, criterion, optimiser and scheduler.
        """
        model = torch.nn.Linear(128, len(self.classes))
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimiser = optim.SGD(model.parameters(), lr=self.training_settings["lr"], momentum=self.training_settings["momentum"], nesterov=self.training_settings["use_nesterov"])
        scheduler = CosineAnnealingLR(optimiser, T_max=self.training_settings["n_epochs"]) # T_max is the number of epochs
        return model, criterion, optimiser, scheduler
    
    def execute(self, version:str, setting:str, embeddings_dir:str) -> None:
        """
        Executes a single training run for the given version and setting, using the embeddings in the given directory.

        Args:
            version (str): The version of the model.
            setting (str): The setting of the model.
            embeddings_dir (str): The directory containing the embeddings of the images.
        
        """
        train_dl, val_dl, test_dl = self.load_data(embeddings_dir=embeddings_dir)

        model, criterion, optimiser, scheduler = self.initialise_components()

        model = train_model(
                            model=model,
                            criterion=criterion,
                            optimiser=optimiser,
                            scheduler=scheduler,
                            train_dl=train_dl,
                            val_dl=val_dl,
                            num_epochs=self.training_settings["n_epochs"],
                            classes=self.classes,
                            device=self.device,
                            print_interval=100
                            )
    
        # Save model
        split_dir = embeddings_dir.split("/")
        iterations_and_b_setting = split_dir[-1]
        MODEL_SAVE_DIR = "linear_eval_models"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_path = f"{MODEL_SAVE_DIR}/linear_{version}_{setting}_{iterations_and_b_setting}.pth"
        torch.save(model.state_dict(), model_path)

        # Test model
        saved_model = torch.nn.Linear(128, len(self.classes))
        saved_model.load_state_dict(torch.load(model_path))
        saved_model.to(self.device)

        true_positive, false_positive, false_negative, total = test_model(
                                                                        model=saved_model,
                                                                        test_dl=test_dl,
                                                                        classes=self.classes,
                                                                        device=self.device
                                                                        )
        precision, recall, f1_score, accuracy = calculate_metrics(
                                                            true_positive=true_positive, 
                                                            false_positive=false_positive, 
                                                            false_negative=false_negative, 
                                                            total=total,
                                                            classes=self.classes,
                                                            )
        total_accuracy = sum(true_positive.values()) / sum(total.values())

        completed_dict = {
                        "model": saved_model.state_dict(),
                        "metrics": {
                                    "precision": precision,
                                    "recall": recall,
                                    "f1_score": f1_score,
                                    "accuracy": accuracy,
                                    "total_accuracy": total_accuracy
                                    }
                        }
        print("Completed training for version:", version, "and setting:", setting)

        # Save the completed dict in the same location (overwrite)
        with open(f"{MODEL_SAVE_DIR}/completed_{version}_{setting}_{iterations_and_b_setting}.pth", "wb") as f:
            torch.save(completed_dict, f)