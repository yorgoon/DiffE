# Diff-E
Decoding EEG signals for imagined speech is a challenging task due to the high-dimensional nature of the data and low signal-to-noise ratio. In recent years, denoising diffusion probabilistic models (DDPMs) have emerged as promising approaches for representation learning in various domains. Our study proposes a novel method for decoding EEG signals for imagined speech using DDPMs and a conditional autoencoder named Diff-E. Results indicate that Diff-E significantly improves the accuracy of decoding EEG signals for imagined speech compared to traditional machine learning techniques and baseline models. Our findings suggest that DDPMs can be an effective tool for EEG signal decoding, with potential implications for the development of brain-computer interfaces that enable communication through imagined speech.

This work is submitted to [Interspeech 2023](https://www.interspeech2023.org/)
## Code
The code implementation is based on GitHub repositories [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) and [Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)
## Todo
- [x] Item 1: Streamline the code
- [ ] Item 2: Document the code
- [ ] Item 3: Test on public datasets
- [ ] Item 4: Experiment on adding temporal convolutional layers
