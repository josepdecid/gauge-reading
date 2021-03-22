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

## Code Structure

The core code is separated into different packages according to the given functionalities.

- Managers
- Misc
- Models
- Transforms

## Thinking process

Some of the thinking process belongs to the Code Structure section, as defining the program flow and components already requires from a thinking process.
However, in this section I want to go into details about the thinking process that involves the decisions about the loss function, network architecture, data augmentations...

