# HiDDeN
Pytorch implementation of paper "HiDDeN: Hiding Data With Deep Networks" by Jiren Zhu*, Russell Kaplan*, Justin Johnson, and Li Fei-Fei: https://arxiv.org/abs/1807.09937  
*: These authors contributed equally

The authors have Lua+Torch implementation here: https://github.com/jirenz/HiDDeN

The original implementation can be found here: https://github.com/ando-khachatryan/HiDDeN/tree/master.

## Requirements

You need [Pytorch](https://pytorch.org/) 2.5+ with TorchVision to run this.
The code has been tested with Python 3.10.12+ and runs both on Ubuntu 24.04, Manjaro Linux kernel 6.6.90 and Windows 11.

## Data

We use 10,000 images for training and 1,000 images for testing. Following the original paper, we chose 
those 10,000 + 1,000 images randomly from one of the coco datasets.  http://cocodataset.org/#download

The data directory has the following structure:
```
<data_root>/
  train/
    train_class/
      train_image1.jpg
      train_image2.jpg
      ...
  test/
    test_class/
      test_image1.jpg
      test_image2.jpg
      ...
```

```train_class``` and ```test_class``` folders are so that we can use the standard torchvision data loaders without change.

## Running

You will need to install the requirements, then run 
```
python main.py new --name <experiment_name> --data-dir <data_root> --batch-size <b> 
```
If you want to continue from a training run, use 
```
python main.py continue --folder <incomplete_run_folder>
```
There are additional parameters for main.py. Use
```
python main.py --help
```
to see the description of all the parameters.
Each run creates a folder in ./runs/<experiment_name date-and-time> and stores all the information about the run in there.

If you want to test a trained model, run ``` test_model.py ```:

```
python3 test_model.py --options-file "..." --checkpoint-file "..." --source-image "dataset/test/test/*.jpg"

```
Run ```python3 test_model.py --help ``` for more information about the parameters needed. 


### Running with Noise Layers
You can specify noise layers configuration. To do so, use the ```--noise``` switch, following by configuration of noise layer or layers.
For instance, the command 
```
python main.py new --name 'combined-noise' --data-dir /data/ --batch-size 12 --noise  'crop((0.2,0.3),(0.4,0.5))+cropout((0
.11,0.22),(0.33,0.44))+dropout(0.2,0.3)+jpeg()'
```
runs the training with the following noise layers applied to each watermarked image: crop, then cropout, then dropout, then jpeg compression. The parameters of the layers are explained below. **It is important to use the quotes around the noise configuration. Also, avoid redundant spaces** If you want to stack several noise layers, specify them using + in the noise configuration, as shown in the example. 


### Noise Layer paremeters
* _Crop((height_min,height_max),(width_min,width_max))_, where **_(height_min,height_max)_** is a range from which we draw a random number and keep that fraction of the height of the original image. **_(width_min,width_max)_** controls the same for the width of the image. 
Put it another way, given an image with dimensions **_H x W,_** the Crop() will randomly crop this into dimensions **_H' x W'_**, where **_H'/H_** is in the range **_(height_min,height_max)_**, and **_W'/W_** is in the range **_(width_min,width_max)_**. In the paper, the authors use a single parameter **_p_** which shows the ratio **_(H' * W')/ (H * W)_**, i.e. the ratio between the cropped image and the original. In our setting, you can obtain the appropriate **_p_** by picking **_height_min_**, **_height_max_**  **_width_min_**, **_width_max_** to be all equal to **_sqrt(p)_**
*  _Cropout((height_min,height_max), (width_min,width_max))_, the parameters have the same meaning as in case of _Crop_. 
* _Dropout(keep_min, keep_max)_ : where the ratio of the pixels to keep from the watermarked image, **_keep_ratio_**, is drawn uniformly from the range **_(keep_min,keep_max)_**.
* _Resize(keep_min, keep_max)_, where the resize ratio is drawn uniformly from the range **_(keep_min, keep_max)_**. This ratio applies to both dimensions. For instance, of we have Resize(0.7, 0.9), and we randomly draw the number 0.8 for a particular image, then the resulting image will have the dimensions (H * 0.8, W * 0.8).
* _Jpeg_ does not have any parameters. 


## Setup
We try to follow the experimental setup of the original paper as closely as possibly.
We train the network on 10,000 random images from [COCO dataset](http://cocodataset.org/#home). We use 200-400 epochs for training and testing.
The testing is on 1,000 images. During training, we take randomly positioned center crops of the images. This makes sure that there is very low chance the network will see the exact same cropped image during training. For testing, we take center crops which are not random, therefore we can exactly compare metrics from one epoch to another. 

Due to random cropping, we observed no overfitting, and our train and testing metrics (mean square error, binary cross-entropy) were extremely close. 

When measuring the decoder accuracy, we do not use error-correcting codes like in the paper. We take the decoder output, clip it to range [0, 1], then round it up. We also report mean square error of the decoder for consistency with the paper. 

