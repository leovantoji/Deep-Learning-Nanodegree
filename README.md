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

![endpointprogram_2]()
- The **HTTP request** that's sent from your application to your **model** is composed of four parts:
  - **Endpoint**: in the form of a URL, which is commonly known as a web address.
  - **HTTP method**: for the purposes of **deployment**, our application will use the **POST** method only.
  
  ![httpmethods]()
  - **HTTP Headers**: contains additional information, like the format of the data within the message, that's passed to the *receiving* program.
  - **Message (Data or Body)**: for **deployment**, message contains the *user's data* which is input into the model.
- The **HTTP response** sent from your model to your application is composed of three parts:
  - **HTTP Status Code**: If the model sucessfully received and processed the *user's data* that was sent in the **message**, the status code should start with a 2, like *200*.
  - **HTTP Headers**: contains additional information, like the format of the data within the **message**, that's passed to the receiving program.
  - **Message (Data or Body)**: What's returned as the *data* within the **message** is the *output* provided by the **model**.
- Following **REST***ful* **API**, the **application** is responsible to:
  - Format the *user's data* in a way that can be easily put into the **HTTP request message** and **used** by the **model**.
  - Translate the *output* from the **HTTP response message** in a way that's easy for the *application user* to understand.
- Notes regarding the *information* included in the **HTTP messages** sent between the **application** and the **model**:
  - Often *user's data* will need to be in a *CSV* or *JSON* format with a specific *ordering* of the data that's dependent upon the **model** used.
  - Often *output* will be returned in *CSV* or *JSON* format with a specific *ordering* of the returned *output* dependent upon the **model** used.
- Both the **model** and the **application** require a *computing environment* so that they can be run and available for use. One way to *create* and *maintain* these *computing environments* is through the use of **containers**. Specifically, the **model** and the **application** can each be run in a **container** *computing environment*. The **containers** are created using a **script** that contains instructions on which software packages, libraries, and other computing attributes are needed in order to run a *software application*, in our case either the **model** or the **application**.
- A **container** can be thought of as a *standardised collection/bundle of software* that is to be *used* for the specific purpose of *running an application*. A common **container** software is *Docker*.
- A *Docker* containers can:
  - Contain **all** types of different software.
  - The structure of a Docker container enables the container to be created, saved, used, and deleted through a set of common tools.
  - The common tool set works with any container regardless of the software the container contains.
- Basic structure of a container:
  - The **underlying computational infrastructure** which can be: a cloud provider's data centre, an on-premise data centre, or even someone's local computer.
  - An **operating system** run on this computational infrastructure, this could be the operating system of your local machine.
  - A **container engine**, which could be the Docker software running on your local computer. The container engine software enables one to create, save, use, and delete containers.
  - Final two layers:
    - **Libraries** and **binaries** required to launch, run, and maintain the next layer.
    - Application layer.

![container_1]()
- The architecture of containers provides the following advantages:
  1. Isolates the application, which **increases** security.
  2. Requires only software needed to run the application, which uses computational resources **more efficiently** and allows for faster application deployment.
  3. Makes application creation, replication, deletion, and maintenance easier and the same across all applications that are deployed using containers.
  4. Provides a simpler and more secure way to replicate, save, and share containers.
- The **container script** is simply the instructions (algorithm) that is used to create a container; for Docker, these container scripts are referred to as **dockerfiles**.

![container_2]()
- Since a container is not a full virtual machine, it's lighter weight on the OS, which means application can run using little overheads. We can update the container script with new dependencies and create new container with updated softwares.
- Characteristics of **Deployment**:
  - **Model Versioning**: One characteristic of deployment is the version of the model that is to be deployed. Besides saving the **model version** as a part of a *model's metadata* in a database, the *deployment platform* should allow one to indicate a deployed **model's version**. This will make it easier to maintain, monitor, and update the *deployed model*.
  - **Model Monitoring**: Another characteristic of deployment is the ability to easily **monitor** your deployed models. Once a model is deployed, you will want to make certain it continues to meet its performance metrics; otherwise, the application may need to be updated with a *better* performing model.
  - **Model Updating and Routing**: The ability to easily **update** your deployed model is another characteristic of deployment. If a deployed model is failing to meet its performance metrics, it's likely you will need to **update** this model. If there's been a fundamental change in the input data for model predictions, you'll want to **update** the model. The *deployment platform* should support **routing** *differing* proportions of *user requests* to the deployed model; to allow *comparison* of performance between the deployed model *variants*. **Routing** in this way allows for a test of a model *performance* as *compared* to other model *variants*.
  - **Model Predictions**: Another characteristic of deployment is the type of **predictions** provided by your deployed model. There are *two common* types of **predictions*:
    - **On-demand predictions**: 
      - Other names: online, real-time, or synchronous predictions.
      - **Low latency** of response to each prediction request.
      - Allowing for **high variability** in request volume.
      - Predictions are returned in the response from the request. Often these requests and responses are done through an API using JSON or XML formatted strings.
      - Each prediction request from the user can contain *one* or *many* requests for predictions. Noting that *many* is limited based upon the *size* of the data sent as the request. Common cloud platforms **on-demand prediction** request size limits can range from *1.5 (ML Engine)* to *5 Megabytes (SageMaker)*.
      - **On-demand predictions** are commonly used to provide customers, users, or employees with real-time, online responses based upon a deployed model.
    - **Batch predictions**:
      - Other names: asynchronous, or batch-based predictions.
      - **High volume** of requests with more *periodic submissions*. Latency won't be an issue.
      - Each batch request will point to specifically *formatted data file* of requests and will return the predictions to a file. Cloud services **require** these files to be stored in the cloud provider's cloud.
      - Cloud services typically have *limits* to how much data they can process with each batch request based upon *limits* they impose on the *size of file* you can store in their cloud storage service. For example, *Amazon's SageMaker* limits batch predictions requests to the size limit they enforce on an object in their S3 storage service.
      - **Batch predictions** are commonly used to help make *business decisions*. For example, imagine a business uses a complex model to predict customer satisfaction across a number of their products and they need these estimates for a weekly report. This would require processing customer data through a **batch prediction** request on a weekly basis.
- Formal definition of **cloud computing**.

[nistcloudcomputing]()
- **Service Models of Cloud Computing**:
  - **Software as a Service (SaaS)**:
  
  [nist-cloudcomputing-servicemodelssaas]()
  - **Platform as a Service (PaaS)**:
  
  [nist-cloudcomputing-servicemodelspaas]()
  - **Infrastructure as a Service (SaaS)**:
  
  [nist-cloudcomputing-servicemodelsiaas]()
- **Deployment Models of Cloud Computing**: Different levels of security are automatically provided with each deployment model. Uderstanding deployment models are important in providing the appropriate level of security on the cloud services you use. The primary way to distinguish between different deployment models is by the group that's provisioning the cloud services. With provisioning, we simply mean the group that's providing the cloud services for their group's use.
  - **Public Cloud**: Most cloud deployment models offered by cloud providers, like Amazon Web Service or Google Cloud Platform, are public clouds. Public clouds are provisioned for use by the general public and they are the least secure of the deployment models. Because public clouds are the least secure, most providers allow customers to enable features that provide a "virtual private cloud" to increase the security on their public clouds.
  - **Community Cloud**: AWS's GovCloud is an example of a community cloud. GovCloud customers are limited to US federal government employees, contractors, and agencies that have been granted access. Community clouds are more secure than public clouds because providers limiting access to community members, and they isolate computing resources used by the community. A community cloud deployment model is provisioned by a specific community of customers from different organisations that have shared interests, like the US federal government in our GovCloud example. Shared interests may include a shared mission, security requirements, policies, and/or compliance considerations.
  - **Private and Hybrid Clouds**: Private cloud deployment model is provisioned for exclusive use by a single organisation. AWS offers a Virtual Private Cloud service, but this isn't considered truly a private cloud unless customers enable additional features that isolate the hardware their cloud services are run on. A private cloud can be physically located on a company's on-site data centre or on a cloud provider's data centre. If the private cloud is provided by a cloud provider, then the private cloud services and infrastructure are always maintained on a private network with hardware and software dedicated solely to the customer's organisation. The hybrid clouds are simply composed of two or three of the different types of deployment models.
- **Essential Characteristics of Cloud Computing**:
  - **On-Demand Self Service**: The essential characteristic on-demand self service means that a customer must be able to automatically provision computing capabilities from the cloud provider without requiring human interaction from the provider. Using our example of cloud storage, on-demand self service is characterised by the ability to log into the cloud storage service and upload, download, or delete the files you have stored within the service at any time without requiring a cloud provider employee to help with these activities.
  - **Broad Network Access**: means that the computing capabilities are made available over the network accessed through standard mechanisms. Your smartphone, tablet, computer use this broad network access to allow you to work remotely, check social media, reply to an email, and generally access the internet. With our example of cloud storage, Broad Network Access is characterised by the ability to access the files you have stored with the cloud provider using any device that is connected to the internet. Noting that the cloud provider is responsible to ensure the path between the network and the data centre.
  - **Resource Pooling**: allows the cloud provider’s computing capabilities to be pooled to serve multiple customers with different virtual resources that are dynamically assigned and re-assigned based upon customer demand. Using our example of cloud storage, Resource Pooling enables the cloud provider to supply storage services to many customers using their data centre to store all their customers’ files. Additionally, the provider’s storage service must be able to handle any customer storing one to many files.
  - **Rapid Elasticity**: enables the cloud provider’s computing capabilities to be elastically provisioned and released to scale rapidly outward and inward proportional to customer demand. This ability to scale rapidly allows the cloud services to appear to have unlimited capabilities that can be used at any time for any quantity of demand. With our example of cloud storage, Rapid Elasticity provides the customer with the ability to store as many or as few files at any time.
  - **Measured Service**: allows the cloud provider’s system to automatically control and optimise resource usage by leveraging the system’s metering capability. Specifically, the cloud provider’s resource usage can be monitored, controlled, and reported, which provides the necessary transparency for appropriately billing customers for cloud service usage. Using our example of cloud storage, Measured Service enables the cloud provider to charge a customer appropriately for storage space they are using to store their files.
- **Capacity Utilisation Graph**:
  - **Wasted Capacity**:
  
  [capacityutilizationcurve-wastedcap4]()
  - **Insufficient Capacity**:
  
  [capacityutilizationcurve-insufficientcap3]()
  - **Dynamic Capacity**:
  
  [capacityutilizationcurve-dynamiccap4]()

## Building a Model using Amazon SageMaker
- **SageMaker session** is a special object that allows you to do things like manage data in S3 and create and train any machine learning models; you can read more about the functions that can be called on a session [here](https://sagemaker.readthedocs.io/en/latest/session.html).
- **SageMaker role**, sometimes called the execution role, is the IAM role that you created when you initialised a notebook instance. The role basically defines how data that your notebook uses/creates will be stored. You can even try printing out the role with `print(role)` to see the details of this creation.
- **S3 (Simple Storage Service)** is a virtual storage solution that is mostly meant for data to be written to few times and read from many times. This is, in some sense, the main workhorse for data storage and transfer when using Amazon services. These are similar to file folders that contain data and metadata about that data, such as the data size, date of upload, author, and so on.
- A **Training job** is used to train a specific estimator. Few items need to be provided before a training job can be executed:
  1. A location on S3 where your training (and possibly validation) data is stored.
  2. A location on S3 where the resulting model will be stored (this data is called the **model artifacts**).
  3. A location of a docker container (certainly this is the case if using a built-in algorithm) to be used for training.
  4. A description of the compute instance that should be used.
- Once all of the abovementioned information is provided, SageMaker will execute the necessary instance (CPU or GPU), load up the necessary docker container and execute it, passing in the location of the training data. After the container finish training the model, the model artifacts will be stored on S3.
- The **transformer** is used to create a transform job and **evaluate a trained model**. The `transform` function takes in the location of some test data and some information about how that test data is formatted.
- In SageMaker, a **model** is a collection of information that describes how to perform inference. For the most part, this comprises two very important pieces:
  1. The container that holds the model inference functionality. For different types of models, this code may be different, but for simpler models and built-in models, this is typically the same container that was used to train the model.
  2. Model artifacts. These are the pieces of data that were created during the training process. For instance, if we were fitting a linear model then the coefficients that were fit would be saved as model artifacts.
- When a model is `fit` using SageMaker, the process is as follows:
  1. A compute instance (basically a server somewhere) is started up with the properties that we specified.
  2. When the compute instance is ready, the code, in the form of a container, that is used to fit the model is loaded and executed. When this code is executed, it is provided access to the training (and possibly validation) data stored on S3.
  3. Once the compute instance has finished fitting the model, the resulting model artifacts are stored on S3, and the compute instance is shut down.

## Deploying and Using a Model
- Using the high level approach makes deploying a trained model pretty straightforward. All we need to do is call the `deploy` method and SageMaker will take care of the rest.
- Sending data to the deployed endpoint and capturing the resulting inference is simple too as SageMaker wraps everything up into the `predict` method, provided we make sure that our data is serialised correctly. In our case, serialising means converting the data structure we wish to send to our endpoint into a string, something that can be transferred using HTTP.
- The cost of a deployed endpoint is based on the length of time that it is running. This means that if you aren't using an endpoint, you **REALLY NEED TO SHUT IT DOWN**.
- Using the low level approach to deploy our model requires us to create an endpoint which will be used to send data to our model and to get inference results.
- In order to create an endpoint in SageMaker, we first need to give SageMaker an endpoint configuration containing various properties we want our endpoint to have. Once we have created the endpoint configuration, we can ask SageMaker to create an endpoint with the properties we want.
- The actual endpoint that is created by SageMaker is a combination of a compute instance running a docker container with the inference code on it and a URL (an endpoint) that data can be sent to and returned from. This URL is used as an interface to the compute instance, which receives data, performs inference using our model and returns the result.
