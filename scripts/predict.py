from keras.models import load_model
from PIL import Image
import os
import numpy as np
import io
from keras.preprocessing.image import img_to_array
def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

model_name = '../trained-models/VGG16-100-0.0001-adam.h5'
model = load_model(model_name)

result = []
for image in os.listdir('./data'):
	with open(os.path.join('./data', image), 'rb') as f:
		image = Image.open(io.BytesIO(f.read()))
		processed_image = prepare_image(image, target=(224, 224))
		preds = model.predict(processed_image)
		pred = np.argmax(preds,axis=1)[0]
		print(pred)
		result.append(pred)


print(result)