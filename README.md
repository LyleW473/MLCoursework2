1. Install requirements using "pip install -r requirements.txt".
2. Run "build_typiclust_dataset.py" and "build_random_dataset.py" to generate the datasets + embeddings.
3. Run "linear_train.ipynb" to train a linear classifier on the generated embeddings" or Run "fully_supervised_train.ipynb" to train a ResNet-18 model on the CIFAR-10 images (after clustering the images).
