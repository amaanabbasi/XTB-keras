# XTB-keras
Identyfying TB from lung X-rays using CNNs. 

![Screenshot from 2019-07-16 12-24-48](https://user-images.githubusercontent.com/30196830/61272452-d1fda380-a7c4-11e9-94e8-36d0d7853ecf.png)

# Dataset 
Dataset is obtained from [here](https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities) which consisted of 2 differnet set of scans, ChinaSet and Montgomery.

# Navigation
* Notebooks: It contains the code for the training part of the model with the various parameters used in the model. This is the entry point where you can start training the model and experiment with different parameters.

* Scripts: contains the contains `.py` files.

* web-app: Contains code for the web-application, A prototype to give a overview on what the final application might look like.

# Requirements
* Python(3+)
* keras
* Jupyter Notebook
* Flask(to run web-app locally, optional)
* The web application is live at [Heroku](https://tb-classifier.herokuapp.com/)

# Usage
`cd Notebooks`

`jupyter notebook`

# Model Performace 

| Model No. | Description | epochs | learning rate | specificity | sensitivity | accuracy |
|---    |---          |---     |---            |---          |---          |---       |
| #1 Vgg16 | trained on last 10 layers | 50 | 0.001 | * | * | 0.79 |
| #2 Vgg16 | trained on last 5 layers | 100 | 0.0001 | 0.766 | 0.85 | 0.789 |
| #3 Vgg16 | trained on all layers | 100 | 0.0001 | 0 | 1 | 0.5 |

### #2 Vgg16

![VGG16(m)-accuracy-100-epochs](https://user-images.githubusercontent.com/30196830/61233908-4f86cc80-a74f-11e9-818c-1aa11bbd51fb.png)

![VGG16(m)-loss-100-epochs](https://user-images.githubusercontent.com/30196830/61233910-531a5380-a74f-11e9-8ea3-3e870bbe3ef2.png)



# Explainable Model

LIME (Local Interpretable Model-Agnostic Explainations) is method to get explaination about the model on why it is making certain decesion it's making.
It will help us improve upon the model and through this we can get an idea how to proceed further and at any point of time we can rely on it for feedback 
and based on that we can take better decisions to get in the right direction.

Below is an image of a X-ray which is classified as positive, and the black outline tells it is the major contributing factor in the decesion by the classifier.
![Screenshot from 2020-03-29 13-21-19](https://user-images.githubusercontent.com/30196830/77848801-02675f80-71e5-11ea-9b77-094d4c47d6db.png)



