import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle
import pandas as pd
import os
LR=1e-3

df=pd.read_csv(r'train.csv')
MODEL_NAME='ageprediction.model'.format(LR,'2conv-basic-video')

#print(df.head())
TRAIN_DIR=r"Train"
IMG_SIZE=32
TEST_DIR=r"Test"
def return_class_from_img(df,img):
	for x,i in enumerate(df['ID']):
		if str(i)==str(img):
			
			age_label= df['Class'][x]

			if age_label=='YOUNG':
				return[1,0,0]
			elif age_label=='MIDDLE':
				return[0,1,0]
			elif age_label=='OLD':
				return [0,0,1]
			
#print(return_class_from_img(df,'377.jpg'))
def train_images():
	training_data=[]
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label=return_class_from_img(df,str(img))
		path=os.path.join(TRAIN_DIR,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img),np.array(label)])
	shuffle(training_data)
	np.save('train_data.npy',training_data)
	return training_data
train_data=train_images()
def test_images():
	testing_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		img_num=img.split('.')[0]
		path =os.path.join(TEST_DIR,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img),np.array(img_num)])
	np.save('testing_data.npy',testing_data)
	return testing_data


import tflearn
from tflearn.layers.conv import max_pool_2d,conv_2d
from tflearn.layers.core import input_data,fully_connected,dropout
from tflearn.layers.estimator import regression


convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')

convnet=conv_2d(convnet,32,3,activation='relu')
convnet=max_pool_2d(convnet,3)
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=max_pool_2d(convnet,3)
convnet=conv_2d(convnet,32,3,activation='relu')
convnet=max_pool_2d(convnet,3)
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=max_pool_2d(convnet,3)
convnet=conv_2d(convnet,32,3,activation='relu')
convnet=max_pool_2d(convnet,3)
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=max_pool_2d(convnet,3)
convnet=conv_2d(convnet,32,3,activation='relu')
convnet=max_pool_2d(convnet,3)
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=max_pool_2d(convnet,3)



convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)


convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')

'''if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('Model Loaded')'''


train=train_data[:-500]
test=train_data[-500:]
X=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y=[i[1] for i in train]

test_x=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)

test_y=[i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id='MODEL_NAME')

import tensorflow as tf
tf.reset_default_graph()

model.save(MODEL_NAME)
'''import matplotlib.pyplot as plt

fig=plt.figure()
for num,data in enumerate(test_data[:20]):
	img_num=data[1]
	img_data=data[0]
	y=fig.add_subplot(4,5,num+1)
	orig=img_data
	data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
	model_out=model.predict([data])[0]
	if np.argmax(model_out)==0:
		str_label='YOUNG'

	elif np.argmax(model_out)==1:
		str_label='MIDDLE'

	elif np.argmax(model_out)==2:
		str_label='OLD'
	y.imshow(orig,cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()'''
test_data=test_images()
with open('submission-file.csv','w') as f:
	f.write('ID,Class\n')
with open('submission-file.csv','a') as f:
	for data in tqdm(test_data):
		img_num=data[1]
		img_data=data[0]
		#y=fig.add_subplot(3,4,num+1)
		orig=img_data
		data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
		model_out=model.predict([data])[0]
		#print(model_out)
		if np.argmax(model_out)==0:
			str_label='YOUNG'

		elif np.argmax(model_out)==1:
			str_label='MIDDLE'

		elif np.argmax(model_out)==2:
			str_label='OLD'
		f.write(f'{str(img_num)}.jpg,{str_label}\n')






