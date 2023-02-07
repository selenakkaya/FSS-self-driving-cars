## Federated Semantic Segmentation
### Advanced Machine Learning - Polito
**Federated semantic segmentation** is a technique that **allows multiple participants, each with their own data, to train a semantic segmentation model together without sharing their data with one another.** This is done by **training a global model on each participant's local data and then aggregating the resulting models to create a final, global model.**

**The goal of federated learning is to allow a group of participants to collaboratively train a machine learning model without having to share their data**. This is particularly useful in situations where the data is sensitive or private, and sharing it would be inappropriate.

In **semantic segmentation**, the task is to assign a label to each pixel in an image. This is a challenging problem, as the labels for different pixels may vary widely and depend on context. 

To achieve accurate semantic segmentation, a large amount of labeled training data is required. **Federated semantic segmentation** allows multiple participants with different data to train a single model, which can lead to better performance than if each participant were to train their own model independently.

A common approach in federated semantic segmentation is to use **federated averaging**, which is a method for **aggregating models trained by multiple participants**. In this method, **each participant trains a model on their local data and then sends the parameters of their model to a central server.** **The central server then averages the parameters of all the models it has received, and sends the resulting model back to each participant. This process is repeated until the model converges.**

There are many challenges to this approach, including but not limited to, non-IID data, distribution shift and communication costs, but researchers have come up with solutions to tackle these challenges.

### Federated Semantic Segmentation for self-driving cars

Federated semantic segmentation for self-driving cars is a technique that **allows multiple vehicles, each with their own sensor data, to train a semantic segmentation model together without sharing their data with one another.** This allows the vehicles to collaboratively improve the accuracy of the semantic segmentation model without compromising the privacy of the data collected by each vehicle.

Self-driving cars rely heavily on semantic segmentation models to understand the environment around them. These models are trained to identify and label different objects in the vehicle's sensor data, such as cars, pedestrians, and traffic signs. The accuracy of these models is crucial for the safety of the vehicle and its passengers, as well as other road users.

**The vehicles could send the local updates (resulting from the local training process) to a central server which updates the global model. This global model then could be sent back to the cars for the next round of training. The process continues until the model reaches convergence or satisfaction.**
