# Evaluation of Deep Convolutional GenerativeAdversarial Networks for data augmentation ofchest X-ray images

## Abstract

Medical image datasets are usually imbalanced, due to high costs of obtaining the data and time-consuming annotations. Training deep neural network model on such datasets to accurately classify the medical condition does not yield desired results and often over-fits the data on majority class samples. In order to address this issue, data augmentation is often performed on training data by position augmentation techniques such as scaling, cropping, flipping, padding, rotation, translation, affine transformation, and color augmentation techniques such as brightness , contrast, saturation, and hue to increase the dataset sizes. These augmentation techniques are not guaranteed to be advantageous in domains with limited data, especially medical image data, and could lead to further overfitting. In this work, we performed data augmentation on Chest X-rays dataset through generative modeling (deep convolutional generative adversarial network) which creates artificial instances retaining similar characteristics to the original data and evaluation of the model resulted in Fréchet Distance of Inception (FID) of 1.289.

To calculate FID: pip install pytorch-fid or download https://github.com/mseitzer/pytorch-fid

Modify the file fid_score.py to update the following lines:

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename).convert('RGB').resize((299,299)), dtype=np.uint8)[..., :3]
    
python console: python -m pytorch_fid path/to/dataset1 path/to/dataset2 --dims 768
jupyter notebook: %run -m pytorch_fid path/to/dataset1 path/to/dataset2 --dims 768



