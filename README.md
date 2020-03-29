# XTB-keras

Identyfying TB from lung X-rays using CNNs. 


# Navigation

* Notebooks: It contains the code for the training part of the model with the various parameters used in the model. This is the entry point where you can start training your own models and experiment with different parameters.
* web-app: Contains code for the web-application, A prototype to give a overview on what the final application might look like.

# Requirements

* Python(3+)
* keras
* Flask(to run web-app locally)

* The web application is live at [........](https://tb-classifier.herokuapp.com/)

# Explainable Model

LIME (Local Interpretable Model-Agnostic Explainations) is method to get explaination about the model on why it is making certain decesion it's making.
It will help us improve upon the model and through this we can get an idea how to proceed further and at any point of time we can rely on it for feedback 
and based on that we can take better decisions to get in the right direction.

Below is an image of a X-ray which is classified as positive, and the black outline tells it is the major contributing factor in the decesion by the classifier.



