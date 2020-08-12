# @Author: Jan Brejcha <janbrejcha>
# @Email:  brejcha@adobe.com, ibrejcha@fit.vutbr.cz, brejchaja@gmail.com
# @Project: ImmersiveTripReports 2017-2018
# AdobePatentID="P7840-US"

# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# last modified date: Dec. 27, 2017, migrating everything to python36 and latest pytorch and torchvision

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image

class Places365Classifier(object):
	"""This class is heavily based on:
	https://github.com/CSAILVision/places365/blob/master/run_placesCNN_unified.py"""

	def __init__(self):

		# load the labels
		self.classes, self.labels_IO, self.labels_attribute, self.labels_attribute_clusters, self.W_attribute = self.load_labels()

		# load the model
		self.features_blobs = []
		self.model = self.load_model()

		# load the transformer
		self.tf = self.returnTF() # image transformer

		# get the softmax weight
		self.params = list(self.model.parameters())
		self.weight_softmax = self.params[-2].data.numpy()
		self.weight_softmax[self.weight_softmax<0] = 0

	def load_labels(self):
		# prepare all the labels
		# scene category relevant
		file_name_category = 'categories_places365.txt'
		if not os.access(file_name_category, os.W_OK):
			synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
			os.system('wget ' + synset_url)
		classes = list()
		with open(file_name_category) as class_file:
			for line in class_file:
				classes.append(line.strip().split(' ')[0][3:])
		classes = tuple(classes)

		# indoor and outdoor relevant
		file_name_IO = 'IO_places365.txt'
		if not os.access(file_name_IO, os.W_OK):
			synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
			os.system('wget ' + synset_url)
		with open(file_name_IO) as f:
			lines = f.readlines()
			labels_IO = []
			for line in lines:
				items = line.rstrip().split()
				labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
		labels_IO = np.array(labels_IO)

		# scene attribute relevant
		file_name_attribute = 'labels_sunattribute.txt'
		if not os.access(file_name_attribute, os.W_OK):
			synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
			os.system('wget ' + synset_url)
		with open(file_name_attribute) as f:
			lines = f.readlines()
			labels_attribute = [item.rstrip() for item in lines]
		file_name_W = 'W_sceneattribute_wideresnet18.npy'
		if not os.access(file_name_W, os.W_OK):
			synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
			os.system('wget ' + synset_url)
		W_attribute = np.load(file_name_W)

		#file contains attribute and list of clusters into which this attribute
		#falls, eg., swimming can be either natural and indoor.
		file_name_attribute_clusters = "labels_sunattribute_clusters.txt"
		attribute_clusters = {}
		with open(file_name_attribute_clusters) as f:
			lines = f.readlines()
			for line in lines:
				items = line.rstrip().split()
				label_list = []
				key = items[0]
				for i in range(1, len(items)):
					if items[i].isdigit():
						label_list.append(int(items[i]))
					else:
						key += ' ' + items[i]
				attribute_clusters.update({key:label_list})

		return classes, labels_IO, labels_attribute, attribute_clusters, W_attribute

	def hook_feature(self, module, input, output):
		self.features_blobs.append(np.squeeze(output.data.cpu().numpy()))

	def returnCAM(self, feature_conv, weight_softmax, class_idx):
		# generate the class activation maps upsample to 256x256
		size_upsample = (256, 256)
		nc, h, w = feature_conv.shape
		output_cam = []
		for idx in class_idx:
			cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
			cam = cam.reshape(h, w)
			cam = cam - np.min(cam)
			cam_img = cam / np.max(cam)
			cam_img = np.uint8(255 * cam_img)
			output_cam.append(cv2.resize(cam_img, size_upsample))
		return output_cam

	def returnTF(self):
	# load the image transformer
		tf = trn.Compose([
			trn.Resize((224,224)),
			trn.ToTensor(),
			trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		return tf

	def load_model(self):
		# this model has a last conv feature map as 14x14

		model_file = 'wideresnet18_places365.pth.tar'
		if not os.access(model_file, os.W_OK):
			os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
			os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

		import wideresnet
		model = wideresnet.resnet18(num_classes=365)
		checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
		state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict)
		model.eval()



		# the following is deprecated, everything is migrated to python36

		## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
		#from functools import partial
		#import pickle
		#pickle.load = partial(pickle.load, encoding="latin1")
		#pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		#model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

		model.eval()
		# hook the feature extractor
		features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
		for name in features_names:
			model._modules.get(name).register_forward_hook(self.hook_feature)
		return model

	def isOutdoor(self, io_image):
		"""	Returns true, if the image is outdoor.

			@param io_image first score in tuple returned by classifyImage()
		"""
		return io_image > 0.5

	def isNatural(self, img_attibutes):
		""" Returns true, if the image is considered to be natural given the
		 	image attributes and their correlation coefficients.
			Definition on what attributes are considered natural is defined in
			labels_sunattribute_clusters.txt. 1 stands for natural, 0 for
			everything else.

			@param img_attributes dictionary with attribute id and its
			correlation coefficient - third element of the tuple returned by
			classifyImage().
		"""
		score = 0
		for att in img_attibutes:
			att_coef = img_attibutes[att]
			label = self.labels_attribute[att]
			clusters = self.labels_attribute_clusters[label]
			#print(label + ": " + str(att_coef))
			if att_coef > 0:
				if 1 in clusters:
					#is natural
					score += att_coef
				else:
					score -= att_coef
		#print("natural score: " + str(score))
		if score > 0:
			return True
		return False


	def classifyImage(self, img_path):
		""" Classifies the image. Estimates score, that image is indoor or
			outdoor, image classes with probabilities, and image attributes with
			correlation coefficients.

			@param img_path path of the image to be classified.
		"""
		# load the model
		self.features_blobs = []
		self.model = self.load_model()

		img = Image.open(img_path)
		img = img.convert("RGB")
		input_img = V(self.tf(img).unsqueeze(0), volatile=True)

		# forward pass
		logit = self.model.forward(input_img)
		h_x = F.softmax(logit, 1).data.squeeze()
		probs, idx = h_x.sort(0, True)

		# output the IO prediction
		io_image = np.mean(self.labels_IO[idx[:10].numpy()]) # vote for the indoor or outdoor

		# output the prediction of scene category
		img_classes = {}
		for i in range(0, 5):
			#print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))
			img_classes.update({idx[i]:probs[i]})

		# output the scene attributes
		responses_attribute = self.W_attribute.dot(self.features_blobs[1])
		idx_a = np.argsort(responses_attribute)
		#print('--SCENE ATTRIBUTES:')

		img_attributes = {}
		for i in range(-1,-10,-1):
			#print(self.labels_attribute[idx_a[i]] + " - " + str(responses_attribute[idx_a[i]]))
			img_attributes.update({idx_a[i]: responses_attribute[idx_a[i]]})

		return (io_image, img_classes, img_attributes)
		# generate class activation mapping
		#print('Class activation map is saved as cam.jpg')
		#CAMs = self.returnCAM(self.features_blobs[0], self.weight_softmax, [idx[0]])

		# render the CAM and output
		#img = cv2.imread(img_path)
		#height, width, _ = img.shape
		#heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
		#result = heatmap * 0.4 + img * 0.5
		#v2.imwrite('cam.jpg', result)
