# Gauge Reading

## How to work with it?

For this project I have used PyTorch 1.8.0 and some other libraries that are listed in the `requirements.txt` file.
If you need to work with the project, create a virtual environment and install the listed requirements to ensure the reproducibility with the same versions.

All the Python source codes can be found inside the `src` folder. You can find the two main scripts in the root folder.
With `train.py` you can train a model by specifying the configuration in the program arguments, such as the number of epochs, batch size, dataset folder, random seed or previous checkpoint to resume training or do fine-tuning.

As an example, we can train a model with the following command, that will train it over the datasets folder data for 50 epochs with a batch size of 8 and a fixed random seed of 42.

```bash
$ python src/train.py datasets --epochs 50 --bs 8 --seed 42  
```

The other script, `test.py` allows to evaluate the model for a given checkpoint, showing the images of the given dataset with the associated prediction.
The command to execute it, is similar to the train one:

```bash
$ python src/test.py datasets/validation --checkpoint_path checkpoints/1/9.pt --bs 8
```

Please, refer to the `--help` flag to obtain further usage information for both scripts.

## Code Structure

The core code is separated into different packages according to the given functionalities.
Even that some packages may be unnecessary for this small project (e.g. models),
I structured it in a way that it would be easy to keep it organized as the project keeps growing.

#### Managers
In this package we have the main classes that manage the model training process, dataset generation and criteria functions to use as  loss functions.

#### Models
The idea of this package is to contain all the possible custom models to train them for our task, or (as it is now), to add some helpers to create and tune some models from the torchvision zoo.

#### Transforms
In transforms, I have created all the custom transformations for data augmentation techniques, as well as some common utility functions that are used on those.

#### Misc
Some general scripts used for data exploration, downloading backgrounds from online images and util functions for displaying images.

## Thinking process
Part of the thinking process belongs to the Code Structure section, as defining the program flow and components already requires from a thinking process.
However, in this section I want to go into details about the thinking process that involves the decisions about the loss function, network architecture, data augmentations...

### Dataset
A first analysis of the given data shows us two clear things. The dataset is not large enough and the training data samples are synthetic images (or manually modified) as those do not have a background and there is only the gauge itself in the foreground.
Therefore, the models trained with the training data will not be able to generalize well for real photos.

Hence, we must apply some data augmentation tricks to add variety to the training images. I opt to work with on-the-fly data augmentation, the common approach that torch and torchvision provide,
as it potentially creates a larger number of variations without having to modify the original dataset only at the cost of having to compute every new sample during the training process.

#### Data Augmentations
There are some common already implemented data augmentation techniques from `torchvision` that I should have used, such as `RandomResizedCrop`, `RandomPerspective`, `RandomHorizontalFlip`, with some
slight modifications such as transforming the bounding box along with the image transformation. Another idea was to build somehow a simulator environment to generate a higher number of gauge samples
with a higher diversity of situations such as different angles, illumination, background... but it is a difficult thing to setup in only one week.

Instead of that, I have decided to focus in the implementation of some (somehow) simple but effective new transformations.
The three transformations listed below, assume that the image have an alpha channel that allows to separate the gauge in the foreground from the background. 

##### Resize
As the name states, this step resizes the gauge sub-image by a given `resize_factor`, as well as recalculating the bounding box of the gauge. This step is not properly a data augmentation technique, but a pre-processing step.
The idea behind it is to reduce the dimensionality of the input data, not mainly to try to overcome a typical curse of dimensionality problem, but to improve the performance of the training process, allowing to use a much larger batch size.
I have used this step because my hardware resources were so limited and with the original size of the images and a small model such as ResNet18, I was barely able to pass two elements on the same batch, resulting sometimes in a CUDA out of memory problem.  

##### Random Gauge Offset
This step consists in randomly cropping the gauge sub-image uniformly between a `min_cropping_factor` and a `max_cropping_factor`. 
The cropped result, is then shifted randomly inside the image. The associated bounding box is also recalculated along with the image transformations.

##### Random Background
Finally, we want to overcome the problem of having gauge images with a transparent background. To do so I have the external script `misc/background_downloader.py`, that crawls google images while downloading a given number of those images that match some keywords.
I have thought about possible places where we can find a gauge, so I have extracted images from workshops, car control panels, pipeline systems...

Here we have some of the example retrieved images:

**TODO IMAGES**

This step randomly samples one of these downloaded images and applies some random cropping to it between `min_cropping_factor` and `1`,
to increase the diversity of the resulting data even if we use the same image as the background.

Then, we differentiate the foreground and the background form the gauge image using the alpha channel, substituting transparent background
with the sampled image. The previous example images applied to some gauges from the data look like the following:

**TODO IMAGES**

### Model Architectures


#### Loss Functions