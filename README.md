# Deep-Learning-Nanodegree
## Recurrent Neural Networks (RNNs)
- Recurrent Neural Networks give us a way to incorporate **memory** into our neural networks, and will be critical in analysing sequential data. RNN's are most often associated with **text processing** and **text generation** because of the way sentences are structured as a sequence of words.
- RNNs have a key flaw, as capturing relationships that span more than 8 or 10 steps back is practically impossible. The flaw stems from the **vanishing gradient** problem in which the contribution of information decays geometrically over time. **LSTM** is one option to overcome the **vanishing gradient** problem in RNNs.
- The **feedforward process** comprises the following steps:
  - Calculate the values of the hidden states.
  - Calculate the values of the outputs.
- In the **backpropagation process**, we minimize the network error slightly with each iteration, by adjusting the weights.
- **Activation function** allows the network to represent non-linear relationships between its inputs and outputs. This is extremely important since most real world data is non-linear. The **activation function** contributes to the aforementioned **vanishing gradient** problem. More on this topic can be found [here](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions).
- Reasons for using Mini-batch Gradient Descent are:
  - Reduction of complexity of the training process.
  - Noise reduction.
- Applications of RNNs:
  - Sentiment Analysis.
  - Speech Recognition.
  - Time Series Prediction.
  - Natural Language Processing.
  - Gesture Recognition.
- RNNs are based on the same principles as those behind Feedforward NNs. Two main differences between RNNs and FFNNs.
  - **Sequences** as inputs in the training phase.
  - **Memory** elements.
- Simple RNN is also known as Elman Network.
- In RNNs, the state layer depended on the current inputs, their corresponding weights, the activation function and also on the previous state. In FFNNs, the hidden layer depended only on the current inputs and weights, as well as on an activation function.
- The output vector is calculated in the exact same way as in FFNNs.
  - Linear combination of the inputs to each output node with the corresponding weight matrix *W<sub>y</sub>*: *y<sub>t</sub> = s<sub>t</sub>W<sub>y</sub>*.
  - Softmax function of the same linear combination: *y<sub>t</sub> = σ(s<sub>t</sub>W<sub>y</sub>)*.
- In **Backpropagation Through Time (BPTT)**, we train the network at timestep *t* as well as take into account all of the previous timesteps.
- **LSTM** is invented to solve **Vanishing Gradient** problem. **Gradient Clipping** is used to solve the **Exploding Gradient** problem. The LSTM cell allows a recurrent system to learn over many time steps without the fear of losing information due to the vanishing gradient problem. It is fully differentiable, therefore gives us the option of easily using backpropagation when updating the weights.

## Long Short-Term Memory Networks (LSTMs)
- Other learning materials:
  - [Chris Olah's LSTM post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [Edwin Chen's LSTM post](http://blog.echen.me/2017/05/30/exploring-lstms/)
  - [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=iX5V1WpxxkY)
- An **LSTM** cell comprises:
  - A **Learn Gate** combines the **Short-Term Memory** and the current **Event** and decides to keep the important parts of the combination. Thus, the output of the **Learn Gate** is *N<sub>t</sub> ⊙ i<sub>t</sub>* where:
    - Combine: *N<sub>t</sub> = tanh(W<sub>n</sub>\[STM<sub>t-1</sub>, E<sub>t</sub>\] + b<sub>n</sub>)*
    - Ignore: *i<sub>t</sub> = σ(W<sub>i</sub>\[STM<sub>t-1</sub>, E<sub>t</sub>\] + b<sub>i</sub>)*    
  - A **Forget Gate** takes the **Long-Term Memory** and decides which parts of LTM to remember and to forget. Thus, the output of the **Forget Gate** is *LTM<sub>t-1</sub> ⊙ f<sub>t</sub>* where:
    - Forget Factor: *f<sub>t</sub> = σ(W<sub>f</sub>\[STM<sub>t-1</sub>, E<sub>t</sub>\] + b<sub>f</sub>)*
  - A **Remember Gate** combines the outputs of the **Learn Gate** and **Forget Gate** into a new **Long-Term Memory**. Thus, the output of the **Remember Gate** is *LTM<sub>t-1</sub> ⊙ f<sub>t</sub> + N<sub>t</sub> ⊙ i<sub>t</sub>*.
  - A **Use Gate** combines the outputs of the **Learn Gate** and **Forget Gate** into a new **Short-Term Memory**. Thus, the output of the **Use Gate** is *U<sub>t</sub> ⊙ V<sub>t</sub>* where:
    - *U<sub>t</sub> = tanh(W<sub>u</sub>LTM<sub>t-1</sub>f<sub>t</sub> + b<sub>u</sub>)*
    - *V<sub>t</sub> = σ(W<sub>u</sub>\[STM<sub>t-1</sub>, E<sub>t</sub>\] + b<sub>u</sub>)*

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

## Deploying a Model
- **Machine Learning Workflow** consists of 3 main components:
  - **Explore & Process Data**: 
    - Step 1. Retrieve Data.
    - Step 2. Clean & Explore the Data.
    - Step 3. Prepare / Transform the Data.
  - **Modelling**:
    - Step 4. Develop & Train Model.
    - Step 5. Validate / Evaluate Model.
  - **Deployment**: 
    - Step 6. Deploy to Production.
    - Step 7. Monitor and Update Model & Data.
- **Cloud Computing** can be thought of as transforming an *IT product* into an *IT service*. Generally, think of cloud computing as using an internet connected device to log into a **cloud computing service**, like *Google Drive*, to access an *IT resource*. These *IT resources* are stored in the cloud provider's data centre. Besides cloud storage, other cloud services include: cloud applications, databases, virtual machines, and other services like SageMaker.
- Most of the factors related to choosing cloud computing services, instead of developing on-premise IT resources are related to **time** and **cost**.
- **Benefits** of cloud computing:
  - Reduced investments and proportional costs (providing cost reduction).
  - Increased scalability (providing simplified capacity planning).
  - Increased availability and reliability (providing organisational agility).
- **Risks** of cloud computing:
  - (Potential) Increase in security vulnerabilities.
  - Reduced operational governance control (over cloud resources).
  - Limited portability between cloud providers.
  - Multi-regional compliance and legal issues.
- **Deployment to production** can simply be thought of as a method that integrates a machine learning model into an existing production environment so that the model can be used to make *decisions* or *predictions* based upon *data input* into the model.
- Paths to Deployment (Most to Least commonly used method):
  1. Python model is *converted* into a format that can be used in the production environment.
  2. Model is *coded* in *Predictive Model Markup Language* (PMML) or *Portable Format Analytics* (PFA).
  3. Python model is *recorded* into the programming language of the production environment.
- The 1<sup>st</sup> method is to build a Python model and use *libraries* and *methods* that *convert* the model into code that can be used in the *production environment*. Specifically most popular machine learning software frameworks, like PyTorch, TensorFlow, Sklearn, have methods that will convert Python models into *intermediate standard format*, like ONNX (*Open Neural Network Exchange* format). This *intermediate standard format* then can be *converted* into the software native to the *production environment*.
  - This is the *easiest* and *fastest* way to move a Python model from *modelling* directly to *deployment*.
  - Moving forward, this is typically the way models are *moved* into the *production environment*.
  - Technologies like *containers*, *endpoints*, and *APIs* also help *ease* the *work* required for *deploying* a model into the *production environment*.
- The 2<sup>nd</sup> method is to code the model in PMML or PFA, which are two complementary standards that *simplify* moving predictive models to *deployment* into a *production environment*. The Data Mining Group developed both PMML and PFA to provide vendor-neutral executable model specifications for certain predictive models used by data mining and machine learning. Certain analytic software allows for the direct import of PMML including but *not limited* to IBM, SPSS, R, SAS Base & Enterprise Miner, Apache Spark, Teradata Warehouse Miner, and TIBCO Spotfire.
- The 3<sup>rd</sup> method, which involves recording the Python model into the language of the production environment, often Java or C++, is rarely used anymore because it takes time to recode, test, and validate the model that provides the *same* predictions as the *original* model.
- **Deployment** is **not** commonly included in machine learning curriculum. This likely is associated with the analyst's typical focus on **Exploring and Processing Data** and **Modeling**, and the software developer's focusing more on **Deployment** and the *production environment*. Advances in cloud services, like SageMaker and ML Engine, and deployment technologies, like Containers and REST APIs, allow for analysts to easily take on the responsibilities of deployment.
- The *type* of environment is defined by the **kind** of user who can access the service.
- A *test environment* is one that is used by testers testing an application.
- A *production environment* is one that is used by users using an application.
- The *application* communicates with the *model* through an interface to the model called an **endpoint**. **Interface (endpoint)** allows the *application* to send *user data* to the model and receives output from the *model* based upon that *user data*.
- Communication between the **application** and the **model** is done through the **endpoint (interface)**, where the **endpoint** is an **Application Programming Interface (API)**.
  - An *easy way* to think of an **API**, is as a *set of rules* that enable programs, here the **application** and the **model**, to *communicate* with each other.
  - **RE**presentational **S**tate **T**ransfer, **REST**, architecture is one that uses **HTTP requests** and **responses** to enable communication between the **application** and the **model** through the **endpoint (interface)**.
  - Both **HTTP requests** and **HTTP response** are communications sent between the **application** and the **model**.
- The **HTTP request** that's sent from your application to your **model** is composed of four parts:
  - **Endpoint**: in the form of a URL, which is commonly known as a web address.
  - **HTTP method**: for the purposes of **deployment**, our application will use the **POST** method only.
  
  |HTTP Methods|GET|POST|PUT|DELETE|
  |:-:|:-|:-|:-|:-|
  |Request Action|READ: This request is used to retrieve information. If the information is found, it's sent back as a response|CREATE: This request is used to create new information. Once a new entry is created, it's sent back as the response|UPDATE: This request is used to update information. The PATCH method also updates information, but it's only a partial update with PATCH.|DELETE: This request is used to delete information|
  - 
