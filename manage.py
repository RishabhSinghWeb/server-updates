import random, os, sys, io, time, numpy, PIL.Image
import tensorflow as tf
import numpy as np
from utils.utils_stylegan2 import convert_images_to_uint8
from stylegan2_generator import StyleGan2Generator
import pandas, tensorflow
from tensorflow.keras.models import model_from_json


json_file = open('weights/modelage.json', 'r')
model_json = json_file.read()
json_file.close()
model_age = model_from_json(model_json)
model_age.load_weights("weights/modelage.h5")

json_file = open('weights/modelethnicity.json', 'r')
model_json = json_file.read()
json_file.close()
model_ethnicity = model_from_json(model_json)
model_ethnicity.load_weights("weights/modelethnicity.h5")

json_file = open('weights/modelgender.json', 'r')
model_json = json_file.read()
json_file.close()
model_gender = model_from_json(model_json)
model_gender.load_weights("weights/modelgender.h5")
gen = StyleGan2Generator(weights='ffhq' , impl='ref', gpu=False)
w_avg = np.load('weights/{}_dlatent_avg.npy'.format('ffhq' ))

while True:
	s=0
	for gender in ['Male','Female']:
		for age in range(5):
			for eth in range(5):
				path = 'static/'+gender+'/'+str(age)+'/'+str(eth)
				if not os.path.exists(path): os.makedirs(path)
				else:
					folder = os.listdir(path)
					if len(folder)>500:
						s+=500
						for file_name in folder[500::]:os.unlink(path+'/'+filename)
					else:s+=len(folder)
	with open('w','w') as f:f.write(str(s))
	for loop_no in range(20000-s):
		dlatents = (gen.mapping_network(np.random.RandomState(np.random.randint(0,300000)).randn(1, 512).astype('float32'))) * ((numpy.random.rand()*0.6)+0.3)
		image = convert_images_to_uint8(gen.synthesis_network(dlatents+np.multiply(np.random.rand(512),0.4)), nchw_to_nhwc=True, uint8_cast=True)
		img = PIL.Image.fromarray(numpy.array(image[0]), 'RGB')
		xx= np.array([([jj[1] for jj in ii[106:1024-104:17]]) for ii in image[0]][183:1024-25:17])
		xx=np.array(xx, dtype="float32").reshape(1,48,48,1)#/255
		m=list(model_ethnicity.predict(xx))
		m=model_ethnicity.predict(xx)
		eth =m.argmax()
		age = round(model_age.predict(xx)[0][0])#//100
		gender = 'Female' if round(model_gender.predict(xx)[0][0]) else "Male"

		if age<13:age=0
		elif age<21:age=1
		elif age<33:age=2
		elif age<50:age=3
		else: age=4
		with open('static/'+gender+"/"+str(age)+'/'+str(eth)+'/'+ str(time.time()) + ".jpg", 'w') as f:
			img.save(f, PIL.Image.registered_extensions()['.jpg'])
