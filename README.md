# *Style-Transfer-Algorithm* implemented in TensorFlow

<img src="./lib/images/content/nyc.jpg" width="50%"  align="right">

This is a TensorFlow implementation of [*A Neural Algorithm of Artistic Style*](https://arxiv.org/pdf/1508.06576v2.pdf) using [total variation denoising](http://eeweb.poly.edu/iselesni/lecture_notes/TVDmm/TVDmm.pdf) as a regularizer. A pretrained [VGG network](https://arxiv.org/pdf/1409.1556.pdf) was used. It is provided [here](https://github.com/machrisaa/tensorflow-vgg) by [machrisaa](https://github.com/machrisaa) on GitHub. The VGG implementation was customized to accomodate the implementation requirements and is of the 19-layer variety.

Using this implementation, it is possible to emulate and achieve the same stylistic results as those in the original paper.

The purpose of this repository is to port the joint [texture-synthesizing](https://arxiv.org/pdf/1505.07376v3.pdf) and [representation-inverting](https://arxiv.org/pdf/1412.0035v1.pdf) stylistic-transfer algorithm to TensorFlow.

## Results

<table style="width:100%">
  <tr>
    <th>Style</th> 
    <th>Result</th>
  </tr>
  <tr>
    <td><img src="./lib/images/style/great-wave-of-kanagawa.jpg" width="100%"></td>
    <td><img src="./lib/images/examples/example_great_wave_of_kanagawa.jpg" width=100%"></td> 
  </tr>
  <tr>
    <td><img src="./lib/images/style/starry-night.jpg" width="100%"></td>
    <td><img src="./lib/images/examples/example_starry-night.jpg" width="100%"></td> 
  </tr>
  <tr>
    <td><img src="./lib/images/style/scream.jpg" width="100%"></td>
    <td><img src="./lib/images/examples/example_scream.jpg" width="100%"></td> 
  </tr>
</table>

## Prerequisites

* [Python 3.5](https://www.python.org/downloads/release/python-350/)
* [TensorFlow](https://www.tensorflow.org/) (>= r0.12)
* [scikit-image](http://scikit-image.org/docs/dev/api/skimage.html)
* [NumPy](http://www.numpy.org/)

## Usage

To stylize an image, run:

```sh
python style_transfer.py path/to/input/image path/to/style/image --output path/to/output/image
```

The default paths for input, style, and output are "photo.jpg", "art.jpg", and "./" respectively.The first two files are supplied in this repository.

## Files

* [style_transfer.py](src/style_transfer.py)

    The main script where all the magic happens. 

* [custom_vgg19.py](src/custom_vgg19.py)
    
    A modified implementation of the VGG 19 network. This particular customization changes the default pooling of max pooling to average pooling, which allows more effective gradient flow.

* [vgg19.npy](https://www.dropbox.com/s/68opci8420g7bcl/vgg19.npy?dl=1)

    The weights used by the VGG network. This file is not in this repository due to its size. You must download it and place in the working directory. The program will complain and ask for you to download it with a supplied link if it does not find it.
    
* [utils.py](src/utils.py)

    Auxiliary routines for parsing images into numpy arrays used in the implementation.
