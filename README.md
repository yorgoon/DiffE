# Diff-E: Diffusion-based Learning for Decoding Imagined Speech EEG
Decoding EEG signals for imagined speech is a challenging task due to the high-dimensional nature of the data and low signal-to-noise ratio. In recent years, denoising diffusion probabilistic models (DDPMs) have emerged as promising approaches for representation learning in various domains. Our study proposes a novel method for decoding EEG signals for imagined speech using DDPMs and a conditional autoencoder named Diff-E. Results indicate that Diff-E significantly improves the accuracy of decoding EEG signals for imagined speech compared to traditional machine learning techniques and baseline models. Our findings suggest that DDPMs can be an effective tool for EEG signal decoding, with potential implications for the development of brain-computer interfaces that enable communication through imagined speech.

This work is submitted to [Interspeech 2023](https://www.interspeech2023.org/)
## EEG Classification with DDPM and Diff-E
The code implementation is based on GitHub repositories [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)

This repository provides an implementation of an EEG classification model using Denoising Diffusion Probabilistic Model (DDPM) and Diffusion-based Encoder (DiffE). The model is designed for classification of EEG signals into one of the 13 classes.

### Main Function Description
The main function of this implementation (train) is responsible for training and evaluating the EEG classification model. The implementation is divided into the following steps:

1. Loading and Preparing Data: The data is loaded using the load_data function, and split into training and testing sets using the get_dataloader function. The batch size and path to the data should be specified.

2. Defining the Model: The model consists of four main components: DDPM, Encoder, Decoder, and Linear Classifier. Their dimensions and parameters should be specified before training.

3. Loss Functions and Optimizers: The implementation uses L1 Loss for training the DDPM and Mean Squared Error Loss for the classification task. RMSprop is used as the optimizer for both DDPM and DiffE, and CyclicLR is employed as the learning rate scheduler.

4. Exponential Moving Average (EMA): EMA is applied to the Linear Classifier to improve its generalization during training.

5. Training and Evaluation: The model is trained for a specified number of epochs. During training, DDPM and DiffE are optimized separately, and their loss functions are combined using a weighting factor (alpha). The model is evaluated on the test set at regular intervals, and the best performance metrics are recorded.

6. Command Line Arguments: The main function accepts command-line arguments for specifying the number of subjects to process and the device to use for training (e.g., 'cuda:0').

### Usage
To train the model, run the following command:
```bash
$ python main.py --num_subjects <number_of_subjects> --device <device_to_use>
```
Replace `<number_of_subjects>` with the number of subjects you wish to process and `<device_to_use>` with the device you want to use for training, such as `'cuda:0'` for the first available GPU.


### Dependencies
To run the code, you will need the following libraries:

- `PyTorch`
- `NumPy`
- `scikit-learn`
- `ema_pytorch`
- `tqdm`
- `argparse`

Make sure to install these libraries before running the code.


## Todo
- [x] Item 1: Streamline the code
- [ ] Item 2: Document the code
- [ ] Item 3: Provide pre-trained models
- [ ] Item 3: Test on public datasets
- [ ] Item 4: Experiment on adding temporal convolutional layers
