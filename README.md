1. Install requirements using "pip install -r requirements.txt".
2. Run "build_typiclust_dataset.py" and "build_random_dataset.py" to generate the datasets + embeddings.
3. Run "linear_train.ipynb" to train a linear classifier on the generated embeddings" or Run "fully_supervised_train.ipynb" to train a ResNet-18 model on the CIFAR-10 images (after clustering the images).


The modification of the algorithm has already been implemented into both training notebooks, it is under the variable "model_names", which selects which embeddings to use, the ResNet embeddings from the original SimCLR model or the embeddings from the custom ViT model.
