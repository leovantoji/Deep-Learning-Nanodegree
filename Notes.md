# Deep-Learning-Nanodegree
## Recurrent Neural Networks (RNNs)
- Recurrent Neural Networks give us a way to incorporate **memory** into our neural networks, and will be critical in analysing sequential data. RNN's are most often associated with **text processing** and **text generation** because of the way sentences are structured as a sequence of words.
- RNNs have a key flaw, as capturing relationships that span more than 8 or 10 steps back is practically impossible. The flaw stems from the **vanishing gradient** problem in which the contribution of information decays geometrically over time. **LSTM** is one option to overcome the **vanishing gradient** problem in RNNs.
- The **feedforward process** comprises the following steps:
  - Calculate the values of the hidden states.
  - Calculate the values of the outputs.
- In the **backpropagation process**, we minimize the network error slightly with each iteration, by adjusting the weights.
- **Activation function** allows the network to represent non-linear relationships between its inputs and outputs. This is extremely important since most real world data is non-linear. The **activation function** contributes to the aforementioned **vanishing gradient** problem. More on this topic can be found [here](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions).


## Generative Adversarial Networks (GANs)
- **GANs** are used to generate realistic data.
- Sample usecases:
  - StackGAN takes a textual representation of something and generates a high resolution photo matching that description.
  - iGAN searches for nearest possible realistic image to transform sketches to pictures.
  - Pix2Pix transforms images of one domain to images of other domain.
  - CycleGAN is extremely good at unsupervised image-to-image translation.
- How do GANs work? [Wikipedia](https://en.wikipedia.org/wiki/Generative_adversarial_network)
  - The **generative network (generator)** generates candidates while the **discriminative network (discriminator)** evaluates them.
  - The generative network learns to map from a latent space to a data distribution of interest, while the discriminative network distinguishes candidates produced by the generator from the true data distribution.
  - The generative network's training objective is to increase the error rate of the discriminative network (i.e., fool the discriminator network by producing novel candidates that the discriminator thinks are not synthesized (are part of the true data distribution).
  - A known dataset serves as the initial training data for the **discriminator**. Training it involves presenting it with samples from the training dataset, until it achieves acceptable accuracy.
  - The **generator** trains based on whether it succeeds in fooling the **discriminator**. Typically, the **generator** is seeded with randomised input that is sampled from a predefined latent space (e.g. a multivariate normal distribution). The choice of the random input noise determines which output comes out of the **generator**. Running the **generator** with many different random input noise values produces many different realistic images. The goal is for the images to be fair samples from the distribution over real data. Thereafter, candidates synthesized by the **generator** are evaluated by the **discriminator**.
  - Backpropagation is applied in both neural networks so that the **generator** produces better images, while the **discriminator** becomes more skilled at flagging synthetic images.
  - The **generator** is typically a deconvolutional neural network, and the **discriminator** is a convolutional neural network.
- Tips for training GANs:
  - **Fully connected architecture** can help with very **simple task** (e.g. generating 28x28 picture of numbers like ones from MNIST dataset). Fully connected architecture consists of matrix multiplication by weight matrices. No convolution. No Recurrence. Both the **discriminator** and the **generator** should have at least **1 hidden layer**. Many activation function work well, but **Leaky ReLUs** are especially popular. **Leaky ReLUs** ensure that the gradient can flow through the entire architecture. This is extremely important for GANs because the only way the **generator** can learn is to **receive a gradient** from the **discriminator**. A popular output choice of the **generator** is the **hyperbolic tangent** activation function (tanh), which means the output will be scaled in the interval **\[-1, 1\]**. The output of the **discriminator** needs to be a probability, and thus, the **sigmoid** activation function is chosen.
  - Use **2 Optimisation Algorithms** to minimise the **loss of the discriminator (d_loss)** and the **loss of the generator (g_loss)**. **Adam** is a good choice for the optimiser. One common mistake is not to use the **numerically stable version of cross entropy** where the loss is computed using the logits. The logits are the values produced by the discriminator right before the sigmoid. One GAN-specific trick is to multiply the zero or one labels by a number that is just a little bit smaller than one so that you replace labels of one with labels of, e.g. 0.9, and keep the zero labels at 0. This is a GAN-specific example of the **label smoothing** strategy used to regularise normal classifiers. It helps the **discriminator** to generalise better and avoid learning to make extreme predictions when extrapolating.
  - **Convolution transpose** should be used to transform a narrow and short feature map to a tall and wide one.
  - **Batch normalisation** should be used in most layers of the network. The DCGAN authors recommend using batch normalisation on every layer except the output layer of the **generator** and the input layer of the **discriminator**. The authors also apply batch normalisation to all the real data in one mini-batch and then apply batch normalisation separately to another mini-batch containing all the generated samples.
