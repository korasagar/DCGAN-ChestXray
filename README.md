[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/evaluation-of-deep-convolutional-generative/medical-image-generation-on-chest-x-ray)](https://paperswithcode.com/sota/medical-image-generation-on-chest-x-ray?p=evaluation-of-deep-convolutional-generative)

Dataset can be downloaded from:
Daniel S Kermany, Michael Goldbaum, Wenjia Cai, Carolina CS Valentim, Huiying Liang, Sally L Baxter,Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, et al. Identifying medical diagnoses and treatablediseases by image-based deep learning.Cell, 172(5):1122–1131, 2018.

To calculate FID: pip install pytorch-fid or download https://github.com/mseitzer/pytorch-fid

Modify the file fid_score.py to update the following lines:

def imread(filename):\
	"""\
	Loads an image file into a (height, width, 3) uint8 ndarray.\
	"""\
    return np.asarray(Image.open(filename).convert('RGB').resize((299,299)), dtype=np.uint8)[..., :3]
    
python console: python -m pytorch_fid path/to/dataset1 path/to/dataset2 --dims 768 \
jupyter notebook: %run -m pytorch_fid path/to/dataset1 path/to/dataset2 --dims 768



