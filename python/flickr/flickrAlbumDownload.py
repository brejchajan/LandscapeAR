#!/usr/bin/env python
# coding=utf-8
#
# @Author: Jan Brejcha <janbrejcha>
# @Email:  brejcha@adobe.com, ibrejcha@fit.vutbr.cz, brejchaja@gmail.com
# @Project: ImmersiveTripReports 2017-2018
# AdobePatentID="P7840-US"
#
# Description
# ============
# A python tool to download albums from Flickr.com from user trips.
# Flickr API does not allow to query the albums directly. One needs to
# find photo IDs that satisfy the query, get user ID of each photo,
# for each user list albums (named photosets in FlickrAPI) containing the given
# photo, and take only albums, which contain the <has_requested_photos /> tag.
# Then, the photos inside the photoset can be listed and finally, downloaded.
# The whole procedure is as follows:
#   1) Search for all photos matching the search query, get their ID.
#   2) For each photo, search for userID and store both.
#   3) For each userID and a photoID, list user's photosets with given userID
#   and photo_ids set to the photoID. Remember the photoset which contain
#   <has_requested_photos /> tag, add it to the found set of photosets for given
#   user.
#   4) For each found photosets, list the photos, and download.
#
# Dependencies
# ============
# Library for flickr api used: https://stuvel.eu/flickrapi
# install with: pip install flickrapi
# argparse
# cPickle
# urllib
# numpy
# matplotlib
# Basemap
# requests
# pyexiftool -- https://github.com/smarnach/pyexiftool, thirdparty/pyexiftool
# pyproj
# pytorch

# libraries and binaries
# wget
# proj

from __future__ import print_function

import flickrapi
import pickle
import sys
import urllib.request
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import requests
#from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

from math import sin, cos, sqrt, atan2, radians
import Places365Classifier
import cv2
import operator

import exiftool
import shutil
import re
import argparse

from FlickrPrefs import FlickrPrefs

prefs = FlickrPrefs()
flickr = prefs.flickr
per_page = 500

def getPhotosets(user_id, photo_id):
	"""Returns the ids of photosets owned by user_id, containing photo with
	photo_id"""

	result_photosets = set()
	photoset_data = {}

	photosets = flickr.photosets.getList(user_id = user_id, photo_ids = photo_id)
	pages = photosets['photosets']['pages']
	for p in range(1, pages + 1):
		photosets = flickr.photosets.getList(user_id = user_id, photo_ids = photo_id, page = p)
		total = photosets['photosets']['total']
		for i in range(0, len(photosets['photosets']['photoset'])):
			photoset = photosets['photosets']['photoset'][i]
			has_req_photos = photoset['has_requested_photos']
			if len(has_req_photos) > 0:
				photoset_id = photoset['id']
				old_len = len(result_photosets)
				result_photosets.add(photoset_id)
				new_len = len(result_photosets)
				if (new_len > old_len):
					#was added into the set, and hence is newl
					photoset_data.update({photoset_id:photoset})
	return result_photosets, photoset_data

def getPhotoset(user_id, photoset_id):
	""" Returns info of a single photoset."""

	result_photosets = set()
	photoset_data = {}

	photosets = flickr.photosets.getInfo(user_id = user_id, photoset_id = photoset_id)
	photoset = photosets['photoset']
	photoset_id = photoset['id']
	result_photosets.add(photoset_id)
	photoset_data.update({photoset_id:photoset})

	return result_photosets, photoset_data

def createSinglePhotoset(output_file, user_id, photoset_id):
	"""Creates a database containing a single photoset."""
	photoset_data = {}
	found_users = {}

	photosets, data = getPhotoset(user_id, photoset_id)
	found_users.update({user_id:photosets})
	photoset_data.update(data)
	with open(output_file, 'wb') as fp:
		pickle.dump({'found_users':found_users, 'photoset_data':photoset_data}, fp)

def searchAndDownloadPhotosInGeoCircleWithCount(lat, lon, radius, count, output_file):
	"""Searches and downloads photos from given geo circleself.
	The most recent photographs are queried in a way to accomodate the <count>
	parameter - starting with the most recent photographs and going backwards
	in time to download at least <count> of photographs. Single query is divided into
	several queries with specified time interval, since Flickr API allows to download
max 4K photographs from a specified location at a time."""
	maxphotos = 4000; #maximum number of photographs to download in one batch
	minphotos = 3000; #minimum number of photographs to download in one batch
	try:
		try:
			photos = flickr.photos.search(lat=lat, lon=lon, radius=radius, privacy_filter=1, per_page = 250)
			total_photos = int(photos['photos']['total'])
			if (count < 0):
				#we want to download as many photographs as available
				count = total_photos
			else:
				#we cannot find more photos than flickr has for this geolocation
				count = min(total_photos, count)
			minphotos = min(minphotos, count)
			print("Maximum retrievable images in this area: " + str(count))
			now = time.mktime(datetime.datetime.today().timetuple())
			hour = 3600.0
			found_count = 0
			downloaded_count = 0
			max_uploaded = now
			min_uploaded = now
			time_correct = True
			while (time_correct and (found_count < total_photos) and (downloaded_count < count)):
				found_batch = False
				currdelta = hour
				max_uploaded = min_uploaded
				min_uploaded = min_uploaded - currdelta
				while (not found_batch):
					#we want to download selected count of outdoor&natural
					#images, but cannot exceed the total number of images
					minphotos = min(minphotos, min(count - downloaded_count, total_photos - found_count))
					photos = flickr.photos.search(lat=lat, lon=lon, radius=radius, privacy_filter=1, per_page = 250, min_upload_date=min_uploaded, max_upload_date=max_uploaded)
					cnt = int(photos['photos']['total'])
					try:
						print("searching batch cnt: " + str(cnt) + ", min_uploaded: " + str(datetime.datetime.fromtimestamp(min_uploaded)) + ", max_uploaded: " + str(datetime.datetime.fromtimestamp(max_uploaded)) + ", currdelta: " + str(currdelta))
						if (cnt < minphotos):
							currdelta = 2*currdelta;
							min_uploaded = min_uploaded - currdelta
						elif (cnt > maxphotos):
							currdelta = currdelta / 2.0
							min_uploaded = min_uploaded + currdelta
						else:
							year = datetime.datetime.fromtimestamp(min_uploaded).year
							if (year < 1500 or year > datetime.datetime.now().year):
								time_correct = False
								break
							found_batch = True;
							print("found batch cnt: " + str(cnt) + ", min_uploaded: " + str(datetime.datetime.fromtimestamp(min_uploaded)) + ", max_uploaded: " + str(datetime.datetime.fromtimestamp(max_uploaded)))
							downloaded_cnt = searchAndDownloadPhotosInGeoCircle(lat, lon, radius, output_file, min_uploaded, max_uploaded)
							print("downloaded (outdoor&natural) count: " + str(downloaded_cnt) + ", out of: " + str(cnt) + " in this batch")
							found_count = found_count + cnt
							downloaded_count = downloaded_count + downloaded_cnt
					except ValueError as ve:
						# we got out of bounds with the min_uploaded and max_uploaded
						time_correct = False
						break


			print("total photos downloaded: " + str(downloaded_count))
		except flickrapi.exceptions.FlickrError as fe:
			  print("Flickr error: ", fe, file = sys.stderr)
	except requests.exceptions.RequestException as re:
		print("Request exception, restarting location download. ", re)
		searchAndDownloadPhotosInGeoCircleWithCount(lat, lon, radius, count, output_file)


# min_uploaded and max_uploaded should be unix timestamps
def searchAndDownloadPhotosInGeoCircle(lat, lon, radius, output_file, min_uploaded = -1, max_uploaded = -1):
	if (radius > 32):
		print("Maximum radius can be, according to FlickrAPI, 32 km at the most. Setting the radius to 32 km.");
		radius = 32
	print("Initializing classifier...")
	classifier = Places365Classifier.Places365Classifier()
	print("Classifier initialized...")
	dataset_path = os.path.dirname(output_file)
	output_dir = os.path.join(dataset_path, "images")
	to_delete_file = os.path.join(dataset_path, "to_delete_imgs.txt")
	to_delete_dir = os.path.join(dataset_path, "to_delete")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	if not os.path.exists(to_delete_dir):
		os.makedirs(to_delete_dir)
	num_downloaded = 0

	#load found photos if already exists
	found_photos = {}
	if os.path.exists(output_file):
		with open(output_file, 'rb') as fp:
			found_photos = pickle.load(fp)
	try:
		try:
			if min_uploaded > 0 and max_uploaded > 0:
				photos = flickr.photos.search(lat=lat, lon=lon, radius=radius, privacy_filter=1, per_page = 250, min_upload_date=min_uploaded, max_upload_date=max_uploaded)
			else:
				photos = flickr.photos.search(lat=lat, lon=lon, radius=radius, privacy_filter=1, per_page = 250)
			pages = int(photos['photos']['pages'])
			print("photos total: " + str(photos['photos']['total']) + ", pages total: " + str(pages))

			for p in range(0, pages):
				print("processing " + str(p) + " page")
				try:
					if min_uploaded > 0 and max_uploaded > 0:
						photos = flickr.photos.search(lat=lat, lon=lon, radius=radius, privacy_filter = 1, per_page = 250, page=p, min_upload_date=min_uploaded, max_upload_date=max_uploaded) #250 per page is max for geo search
					else:
						photos = flickr.photos.search(lat=lat, lon=lon, radius=radius, privacy_filter = 1, per_page = 250, page=p) #250 per page is max for geo search
					num_photos = len(photos['photos']['photo'])
					print(num_photos)
					for i in range(0, num_photos):
						photo_id = photos['photos']['photo'][i]['id']

						photo_server = photos['photos']['photo'][i]['server']
						photo_secret = photos['photos']['photo'][i]['secret']
						photo_farm = photos['photos']['photo'][i]['farm']
						photo_path = downloadPhoto(photo_id, photo_server, photo_secret, photo_farm, output_dir)
						try:
							if (not photo_path == ""):
								#img_info = classifier.classifyImage(photo_path)
								#outdoor = classifier.isOutdoor(img_info[0])
								#natural = classifier.isNatural(img_info[2])
								if (True): #outdoor and natural):
									found_photos.update({photo_id:photos['photos']['photo'][i]})
									num_downloaded = num_downloaded + 1
								else:
									print("To be removed: " + photo_path + ", outdoor: " + str(outdoor) + ", natural: " + str(natural))
									shutil.move(photo_path, os.path.join(to_delete_dir, os.path.basename(photo_path)))
									with open(to_delete_file, "a") as tdf:
										tdf.write(photo_path + " outdoor: " + str(outdoor) + ", natural: " + str(natural) + "\n")
						except IOError:
							print("IOError with image: " + photo_path)
					with open(output_file, 'wb') as fp:
						pickle.dump(found_photos, fp)
				except flickrapi.exceptions.FlickrError as fe:
					print("Flickr error: ", fe, file = sys.stderr)
		except flickrapi.exceptions.FlickrError as fe:
			  print("Flickr error: ", fe, file = sys.stderr)
	except requests.exceptions.RequestException as re:
		print("Request exception, restarting location download. ", re)
		searchAndDownloadPhotosInGeoCircle(lat, lon, radius, output_file)
	return num_downloaded


def searchPhotos(query, output_file):
	"""Searches for photos satisfying given query."""
	try:
		#search for public photos satisfying the query
		try:
			photos = flickr.photos.search(text = query, privacy_filter = 1, per_page = per_page)
			pages = int(photos['photos']['pages'])

			print("photos total: " + str(photos['photos']['total']) + ", pages total: " + str(pages))
			photoset_data = {}
			found_users = {}
			for p in range(1, pages):
				print("processing " + str(p) + " page")
				try:
					photos = flickr.photos.search(text = query, privacy_filter = 1, per_page = per_page, page=p)
					num_photos = len(photos['photos']['photo'])
					for i in range(0, num_photos):
						photo_id = photos['photos']['photo'][i]['id']
						owner_id = photos['photos']['photo'][i]['owner']
						photosets, data = getPhotosets(owner_id, photo_id)

						updated = False
						if owner_id in found_users:
							old_len = len(found_users[owner_id])
							found_users[owner_id].update(photosets)
							new_len = len(found_users[owner_id])
							if (new_len > old_len):
								updated = True

						else:
							found_users.update({owner_id:photosets})
							updated = True
						if updated:
							photoset_data.update(data)

							with open(output_file, 'wb') as fp:
								pickle.dump({'found_users':found_users, 'photoset_data':photoset_data}, fp)

							# write newly found photoset for given owner_id to stdout
							for photoset_id in photosets:
								print("User id: " + owner_id + ", photoset id: " + photoset_id)
				except flickrapi.exceptions.FlickrError as fe:
					print("Flickr error: ", fe, file = sys.stderr)
		except flickrapi.exceptions.FlickrError as fe:
			  print("Flickr error: ", fe, file = sys.stderr)
	except requests.exceptions.RequestException as re:
		print("Request exception, restarting location download. ", re)
		searchPhotos(query, output_file)

def downloadPhoto(photo_id, photo_server, photo_secret, photo_farm, output_dir):
	large_photo_name = photo_id+"_"+photo_secret+"_b.jpg"
	large_url = "https://farm"+str(photo_farm)+".staticflickr.com/"+str(photo_server)+"/"+large_photo_name

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	output_photo = os.path.join(output_dir, large_photo_name)

	if not os.path.isfile(output_photo):
		try:
			urllib.request.urlretrieve(large_url, output_photo)
			print("Downloaded: " + output_photo)
			return output_photo
		except IOError:
			print("Cannot download image: " + large_photo_name)
			return ""

	print("Skipping, already downloaded: " + output_photo)
	return ""

def downloadPhotos(found_photos_file, output_dir = None):
	if not output_dir:
		dataset_path = os.path.dirname(found_photos_file)
		output_dir = os.path.join(dataset_path, "images")
	try:
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		with open(found_photos_file, 'rb') as fp:
			data = pickle.load(fp)
			for user_id in data['found_users']:
				photosets = data['found_users'][user_id]
				for photoset_id in photosets:
					try:
						photos = flickr.photosets.getPhotos(user_id = user_id, photoset_id = photoset_id, per_page = per_page, privacy_filter = 1)
						pages = int(photos['photoset']['pages'])
						for p in range(0, pages):
							print("processing " + str(p) + " page")
							try:
								photos = flickr.photosets.getPhotos(user_id = user_id, photoset_id = photoset_id, per_page = per_page, page = p, privacy_filter = 1)
								num_photos = len(photos['photoset']['photo'])
								for i in range(0, num_photos):
									print(photos['photoset']['photo'][i])
									photo_id = photos['photoset']['photo'][i]['id']
									photo_server = photos['photoset']['photo'][i]['server']
									photo_secret = photos['photoset']['photo'][i]['secret']
									photo_farm = photos['photoset']['photo'][i]['farm']

									output_user_photoset_dir = os.path.join(output_dir, user_id, photoset_id)
									downloadPhoto(photo_id, photo_server, photo_secret, photo_farm, output_user_photoset_dir)

							except flickrapi.exceptions.FlickrError as fe:
								print("Flickr error: ", fe, file = sys.stderr)
					except flickrapi.exceptions.FlickrError as fe:
						print("Flickr error: ", fe, file = sys.stderr)
	except requests.exceptions.RequestException as re:
		print("Request exception, restarting location download. ", re)
		downloadPhotos(found_photos_file, output_dir)

def histoNumAlbumsUser(found_photos_file):
	#histogram of number of albums per user
	albums_cnt = np.array([])
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		for user_id in data['found_users']:
			photosets = data['found_users'][user_id]
			print(photosets)
			albums_cnt = np.append(albums_cnt, len(photosets))
	n, bins, patches = plt.hist(albums_cnt, 20, normed=0, alpha=0.75)
	print(albums_cnt)
	plt.title(r'Histogram of albums per user')
	plt.xlabel('Number of albums')
	plt.ylabel('Number of users')
	#plt.show()


def histoNumPhotosInAlbum(found_photos_file):
	#histogram of number of photos in album
	photos_cnt = np.array([])
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		for user_id in data['found_users']:
			photosets = data['found_users'][user_id]
			for photoset_id in photosets:
				photos_cnt = np.append(photos_cnt, int(data['photoset_data'][photoset_id]['photos']))
	binwidth = 50
	bins = np.arange(min(photos_cnt), max(photos_cnt) + binwidth, binwidth)
	n, bins, patches = plt.hist(photos_cnt, bins=bins, normed=0, alpha=0.75)
	plt.title(r'Histogram of photos in album')
	plt.xlabel('Number of photos, bin width: ' + str(binwidth))
	plt.ylabel('Number of albums')
	#plt.show()

def histoFractionPhotosWithGPSinAlbum(found_photos_file):
	#histogram of fraction of photos with gps in album
	photos_cnt = np.array([])
	photos_gps_cnt = np.array([])
	output_location_file = getLocationFileName(found_photos_file)
	if not os.path.isfile(output_location_file):
		downloadLocationData(found_photos_file)
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		with open(output_location_file, 'rb') as fp:
			location_data = pickle.load(fp)
			for user_id in data['found_users']:
				photosets = data['found_users'][user_id]
				for photoset_id in photosets:

					num_photos = int(data['photoset_data'][photoset_id]['photos'])
					if num_photos == 0:
						continue
					if not photoset_id in location_data:
						continue
					num_photos_gps = len(location_data[photoset_id])
					photos_cnt = np.append(photos_cnt, num_photos - num_photos_gps)
					photos_gps_cnt = np.append(photos_gps_cnt, float(num_photos_gps) / float(num_photos))
	binwidth = 50
	bins = np.arange(0, 1.05, 0.05)
	print(photos_gps_cnt)
	#n, bins, patches = plt.hist(photos_cnt, bins=bins, normed=0, alpha=0.75)
	n, bins, patches = plt.hist(photos_gps_cnt, facecolor='green', bins=bins, normed=0, alpha=0.75)
	plt.title(r'Histogram of fraction of photos in an album with GPS')
	plt.xlabel('Fraction of photos with GPS')
	plt.ylabel('Number of albums')
	#plt.show()

def histoLocationAccuracy(found_photos_file):
	#histogram of location accuracy
	gps_acc = np.array([])
	output_location_file = getLocationFileName(found_photos_file)
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		with open(output_location_file, 'rb') as fp:
			location_data = pickle.load(fp)
			for user_id in data['found_users']:
				photosets = data['found_users'][user_id]
				for photoset_id in photosets:
					if not photoset_id in location_data:
						continue
					#print(location_data[photoset_id])
					for key in location_data[photoset_id]:
						acc = location_data[photoset_id][key]['photo']['location']['accuracy']
						gps_acc = np.append(gps_acc, float(acc))

	n, bins, patches = plt.hist(gps_acc, 16, facecolor='green', normed=1, alpha=0.75)
	plt.title(r'Histogram of GPS accuracies - all photos')
	plt.xlabel('Accuracy class (according to Flickr), 16 is the best')
	plt.ylabel('Fraction of images')
	#plt.show()

def haversineDist(lat1, lon1, lat2, lon2):
	# approximate radius of earth in km

	R = 6373.0

	lat1 = radians(lat1)
	lon1 = radians(lon1)
	lat2 = radians(lat2)
	lon2 = radians(lon2)

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))

	distance = R * c
	return distance


# def plotLocationsOnMap(found_photos_file, fig):
#
#
#
# 	mean_album = np.empty([0,2])
# 	dist_mean_album = np.array([])
# 	dist_std_album = np.array([])
# 	gps = np.empty([0,2])
# 	output_location_file = getLocationFileName(found_photos_file)
# 	photosets_info = {}
# 	with open(found_photos_file, 'rb') as fp:
# 		data = pickle.load(fp)
# 		with open(output_location_file, 'rb') as fp:
# 			location_data = pickle.load(fp)
# 			for user_id in data['found_users']:
# 				photosets = data['found_users'][user_id]
# 				for photoset_id in photosets:
# 					gps_photoset = np.empty([0,2])
# 					if not photoset_id in location_data:
# 						continue
# 					#print(location_data[photoset_id])
# 					for key in location_data[photoset_id]:
# 						lat = location_data[photoset_id][key]['photo']['location']['latitude']
# 						lon = location_data[photoset_id][key]['photo']['location']['longitude']
# 						gps = np.append(gps, [[float(lat), float(lon)]], axis=0)
# 						gps_photoset = np.append(gps_photoset, [[float(lat), float(lon)]], axis=0)
# 					photosets_info.update({photoset_id:gps_photoset})
#
# 	for photoset_id in photosets_info:
# 		if np.shape(photosets_info[photoset_id])[0] > 0:
# 			mi_lat = np.mean(photosets_info[photoset_id][:,0])
# 			mi_lon = np.mean(photosets_info[photoset_id][:,1])
# 			mean_album = np.append(mean_album, [[mi_lat, mi_lon]], axis=0)
# 		else:
# 			mean_album = np.append(mean_album, [[0, 0]], axis=0)
#
# 	# compute distances from mean
# 	idx = 0
# 	for photoset_id in photosets_info:
# 		gps_ps = photosets_info[photoset_id]
# 		photo_cnt = np.shape(gps_ps)[0]
# 		mi_lat = mean_album[idx, 0]
# 		mi_lon = mean_album[idx, 1]
# 		dists = np.array([])
# 		for i in range(0, photo_cnt):
# 			lat = gps_ps[i, 0]
# 			lon = gps_ps[i, 1]
#
# 			dist = haversineDist(lat, lon, mi_lat, mi_lon)
# 			#if (dist <= 1000):
# 			dists = np.append(dists, dist)
# 		std = np.std(dists)
# 		mi = np.mean(dists)
#
# 		mi = 0 if np.isnan(mi) else mi
# 		std = 0 if np.isnan(std) else std
# 		dist_std_album = np.append(dist_std_album, std)
# 		dist_mean_album = np.append(dist_mean_album, mi)
# 		idx = idx + 1
#
# 	print("Total number of albums: " + str(len(photosets_info)))
# 	std_threshold = np.std(dist_mean_album)
# 	dist_mean_album = dist_mean_album[np.where(dist_mean_album < std_threshold)] #we don't want outliers
# 	fig.add_subplot(3,3,4)
# 	n, bins, patches = plt.hist(dist_mean_album, 50, facecolor='green', normed=0, alpha=0.75)
# 	plt.title(r'Histogram of of mean of distances')
# 	plt.xlabel('Mean of distance [km], bin width: ' + str(bins[1] - bins[0]))
# 	plt.ylabel('Albums')
# 	#plt.show()
#
# 	fig.add_subplot(3,3,5)
# 	#histogram of standard deviations of distances from mean
# 	std_threshold = np.std(dist_std_album)
# 	dist_std_album = dist_std_album[np.where(dist_std_album < std_threshold)] #we don't want outliers
# 	n, bins, patches = plt.hist(dist_std_album, 50, facecolor='green', normed=0, alpha=0.75)
# 	plt.title(r'Histogram of of standard deviations')
# 	plt.xlabel('Standard deviation of distance [km], bin width: ' + str(bins[1] - bins[0]))
# 	plt.ylabel('Albums')
# 	#plt.show()
#
#
# 	lat_mean = np.mean(gps[:,0])
# 	lon_mean = np.mean(gps[:,1])
# 	lat_std = np.std(gps[:,0])
# 	lon_std = np.std(gps[:,1])
#
#
# 	fig.add_subplot(3,3,6)
# 	#m = Basemap(width=12000000,height=9000000,projection='lcc',
# 	#        resolution='c',lat_1=lat_mean - lat_std,lat_2=lat_mean + lat_std,lat_0=lat_mean,lon_0=lon_mean)
# 	m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
# 			llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
# 	m.drawmapboundary(fill_color='#99ffff')
# 	m.fillcontinents(color='#cc9966',lake_color='#99ffff')
# 	cmap = plt.cm.get_cmap('hsv', len(photosets_info))
# 	idx = 0
# 	for photoset_id in photosets_info:
# 		gpsp = photosets_info[photoset_id]
# 		x, y = m(gpsp[:,1], gpsp[:,0])
# 		#m.bluemarble()
# 		m.scatter(x,y,50,marker='o',color=cmap(idx),alpha=0.2, zorder=1000, label=str(photoset_id))
# 		idx = idx + 1
# 	#plt.legend()


def histoFracOutdoorAndNatural(found_photos_file):
	cls_file_name = getClassificationFileName(found_photos_file)
	cls_data = ""
	with open(cls_file_name, 'rb') as cfp:
		cls_data = pickle.load(cfp)

	outdoor_natural = np.array([])
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		for user_id in data['found_users']:
			photosets = data['found_users'][user_id]
			for photoset_id in photosets:
				if photoset_id in cls_data:
					on = cls_data[photoset_id]["outdoor_and_natural"]
					outdoor_natural = np.append(outdoor_natural, on)
	n_bins = 50
	n, bins, patches = plt.hist(outdoor_natural, n_bins, cumulative=True, histtype='step', normed=1, alpha=0.75)
	n, bins, patches = plt.hist(outdoor_natural, n_bins, cumulative=-1, histtype='step', normed=1, alpha=0.75)
	plt.title(r'Cumulative distbution of fraction of outdoor and natural photos in an album')
	plt.xlabel('Fraction of images being outdoor and natural')
	plt.ylabel('Number of albums')
	#plt.show()

def histoAttributes(found_photos_file):
	cls_file_name = getClassificationFileName(found_photos_file)
	cls_data = ""
	with open(cls_file_name, 'rb') as cfp:
		cls_data = pickle.load(cfp)

	outdoor_natural = np.array([])
	on_attributes = {} #attributes of photos which are outdoor and natural
	non_attributes = {} #attributes of photos which are not outdoor or natural
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		for user_id in data['found_users']:
			photosets = data['found_users'][user_id]
			for photoset_id in photosets:
				if photoset_id in cls_data:
					on = cls_data[photoset_id]["outdoor_and_natural"]

					attributes = cls_data[photoset_id]["attributes"]
					for idx_a in attributes:
						att_scores = np.array(attributes[idx_a])
						if on:
							addItemsToDict(on_attributes, {idx_a:np.mean(att_scores)})
						else:
							addItemsToDict(non_attributes, {idx_a:np.mean(att_scores)})
	#calculate mean from means per attribute
	on_att_mean = {}
	non_att_mean = {}
	for idx_a in on_attributes:
		mean = np.mean(np.array(on_attributes[idx_a]))
		on_att_mean.update({idx_a:mean})
	for idx_a in non_attributes:
		mean = np.mean(np.array(non_attributes[idx_a]))
		non_att_mean.update({idx_a:mean})
	#sort according to mean value
	sorted_on_att_mean = sorted(on_att_mean.items(), key=operator.itemgetter(1))
	sorted_non_att_mean = sorted(non_att_mean.items(), key=operator.itemgetter(1))

	#create histogram of top-5 attributes with highest mean score
	on_keys = []
	on_labels = []
	on_mean = []
	non_keys = []
	non_labels = []
	non_mean = []
	classifier = Places365Classifier.Places365Classifier()
	N = 5
	for i in range(-1, -N-1, -1):
		on_keys.append(sorted_on_att_mean[i][0])
		on_labels.append(classifier.labels_attribute[sorted_on_att_mean[i][0]])
		on_mean.append(sorted_on_att_mean[i][1])
		non_keys.append(sorted_non_att_mean[i][0])
		non_labels.append(classifier.labels_attribute[sorted_non_att_mean[i][0]])
		non_mean.append(sorted_non_att_mean[i][1])

	for i in range(N-1, -1, -1):
		print(i)
		on_keys.append(sorted_on_att_mean[i][0])
		on_labels.append(classifier.labels_attribute[sorted_on_att_mean[i][0]])
		on_mean.append(sorted_on_att_mean[i][1])
		non_keys.append(sorted_non_att_mean[i][0])
		non_labels.append(classifier.labels_attribute[sorted_non_att_mean[i][0]])
		non_mean.append(sorted_non_att_mean[i][1])

	print(on_mean)
	ind = np.arange(4*N)
	bin_width = 0.35
	plt.xticks(rotation=90)
	ax = plt.gca()
	#ax.set_xticks(ind)
	labels = [val for pair in zip(on_labels, non_labels) for val in pair]
	vals = [val for pair in zip(on_mean, non_mean) for val in pair]
	barlist = ax.bar(ind, vals, bin_width, color = 'deepskyblue', tick_label=labels)
	for i in range(1, 4*N, 2):
		barlist[i].set_color('orange')
	blue_patch = mpatches.Patch(color='deepskyblue', label='outdoor and natural')
	orange_patch = mpatches.Patch(color='orange', label='not outdoor or not natural')
	plt.legend(handles=[blue_patch, orange_patch])
	plt.title(r'Mean attribute score')
	plt.xlabel('Attributes')
	plt.ylabel('Correlation coefficient')


def calculateStatistics(found_photos_file):
	fig = plt.figure()
	fig.add_subplot(3,3,1)
	histoNumAlbumsUser(found_photos_file)

	fig.add_subplot(3,3,2)
	histoNumPhotosInAlbum(found_photos_file)

	fig.add_subplot(3,3,3)
	histoFractionPhotosWithGPSinAlbum(found_photos_file)

	#plotLocationsOnMap(found_photos_file, fig)

	fig.add_subplot(3,3,7)
	histoLocationAccuracy(found_photos_file)

	fig.add_subplot(3,3,8)
	histoFracOutdoorAndNatural(found_photos_file)

	fig.add_subplot(3,3,9)
	histoAttributes(found_photos_file)
	plt.show()


def getLocationFileName(found_photos_file):
	return getSuffixFileName(found_photos_file, "_location.p")

def getExifDataFileName(found_photos_file):
	return getSuffixFileName(found_photos_file, "_exif.p")

def getClassificationFileName(found_photos_file):
	return getSuffixFileName(found_photos_file, "_classification.p")

def getSuffixFileName(file_name, suffix):
	filename, file_extension = os.path.splitext(file_name)
	suffix_file = filename + suffix
	return suffix_file

def downloadLocationDataForSingleDataset(found_photos_file):
	try:
		output_location_file = getLocationFileName(found_photos_file)
		all_location = {}
		if os.path.isfile(output_location_file):
			f = open(output_location_file, 'rb')
			all_location = pickle.load(f)
			append = True
			f.close()
		with open(found_photos_file, 'rb') as fp:
			photos = pickle.load(fp)
			num_photos = len(photos)
			idx = 0
			for i in photos:
				photo_id = photos[i]['id']
				location = getLocationForPhotoId(photo_id)
				if location:
					all_location.update({photo_id:location})
				else:
					print("location not found: " + photo_id)
				if idx % 100 == 0:
					print("Done " + str(idx) + " out of " + str(num_photos))
					with open(output_location_file, 'wb') as fp:
						pickle.dump(all_location, fp)
				idx += 1
			with open(output_location_file, 'wb') as fp:
				pickle.dump(all_location, fp)

	except requests.exceptions.RequestException as re:
		print("Request exception, restarting location download. ", re)
		downloadLocationDataForSingleDataset(found_photos_file)


def getLocationForPhotoId(photo_id):
	try:
		location = flickr.photos.geo.getLocation(photo_id = photo_id)
		if location['stat'] == 'ok':
			return location
	except flickrapi.exceptions.FlickrError as fe:
		print("Flickr error: ", fe, file = sys.stderr)
	return None

def downloadLocationData(found_photos_file):
	try:
		output_location_file = getLocationFileName(found_photos_file)

		append = False
		photoset_location = {}
		if os.path.isfile(output_location_file):
			f = open(output_location_file, 'rb')
			photoset_location = pickle.load(f)
			append = True
			f.close()
		with open(found_photos_file, 'rb') as fp:
			data = pickle.load(fp)
			total_users = len(data['found_users'])
			user_cnt = -1
			for user_id in data['found_users']:
				user_cnt = user_cnt + 1
				print("Done: " + str(float(user_cnt) * 100/total_users) + "%")
				photosets = data['found_users'][user_id]
				for photoset_id in photosets:
					if append:
						if photoset_id in photoset_location:
							#print("Skipping photoset_id: " + photoset_id + " - already in database.")
							continue
					try:
						photos = flickr.photosets.getPhotos(user_id = user_id, photoset_id = photoset_id, per_page = per_page, privacy_filter = 1)
						pages = int(photos['photoset']['pages'])
						photo_location = {}
						for p in range(0, pages):
							#print("photoset: " + photoset_id + ", processing " + str(p) + " page")
							try:
								photos = flickr.photosets.getPhotos(user_id = user_id, photoset_id = photoset_id, per_page = per_page, page = p, privacy_filter = 1)
								num_photos = len(photos['photoset']['photo'])
								for i in range(0, num_photos):
									photo_id = photos['photoset']['photo'][i]['id']
									location = getLocationForPhotoId(photo_id)
									if location:
										photo_location.update({photo_id:location})

							except flickrapi.exceptions.FlickrError as fe:
								print("Flickr error: ", fe, file = sys.stderr)
						photoset_location.update({photoset_id:photo_location})
						with open(output_location_file, 'wb') as fp:
							pickle.dump(photoset_location, fp)
					except flickrapi.exceptions.FlickrError as fe:
						print("Flickr error: ", fe, file = sys.stderr)
	except requests.exceptions.RequestException as re:
		print("Request exception, restarting location download. ", re)
		downloadLocationData(found_photos_file)

def addItemsToDict(d, img_classes):
	""" Appends items for each key of of one dictionary to the list in dictionary
		d corresponding to the same key."""
	for c in img_classes:
		if c not in d:
			d.update({c:[img_classes[c]]})
		else:
			d[c].append(img_classes[c])

def show_images(images, cols = 1, plt_title = None, titles = None):
	"""Display a list of images in a single figure with matplotlib.
	source: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1

	Parameters
	---------
	images: List of np.arrays compatible with plt.imshow.

	cols (Default = 1): Number of columns in figure (number of rows is
						set to np.ceil(n_images/float(cols))).

	titles: List of titles corresponding to each image. Must have
			the same length as titles.
	"""
	assert((titles is None)or (len(images) == len(titles)))
	n_images = len(images)
	if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
	fig = plt.figure()

	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		plt.axis('off')
		a.set_xticklabels([])
		a.set_yticklabels([])
		if image.ndim == 2:
			plt.gray()
		plt.imshow(image)
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.suptitle(plt_title)
	#plt.show()
	return fig

def classifyPhotos(found_photos_file):
	dataset_path = os.path.dirname(found_photos_file)
	classifier = Places365Classifier.Places365Classifier()

	classification_file = getClassificationFileName(found_photos_file)

	photosets_info = {}
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		for user_id in data['found_users']:
			photosets = data['found_users'][user_id]
			for photoset_id in photosets:
				images_info = {}
				image_dir = os.path.join(dataset_path, "images", user_id, photoset_id)
				#iterate the photos
				photoset_classes = {}
				photoset_attributes = {}
				outdoor_and_natural = 0
				total = 0
				if os.path.isdir(image_dir):
					for filename in os.listdir(image_dir):
						img_path = os.path.join(image_dir, filename)
						img_info = classifier.classifyImage(img_path)
						outdoor = classifier.isOutdoor(img_info[0])
						natural = classifier.isNatural(img_info[2])
						images_info.update({'name':filename,'outdoor':outdoor,'natural':natural})
						addItemsToDict(photoset_classes, img_info[1])
						addItemsToDict(photoset_attributes, img_info[2])
						if True: #outdoor and natural:
							outdoor_and_natural +=1
						total += 1
					photoset_info = {'classes':photoset_classes, 'attributes': photoset_attributes, 'images_info':images_info, 'outdoor_and_natural':float(outdoor_and_natural) / float(total)}
					print("photoset: " + photoset_id + ", outdoor and natural: " + str(photoset_info['outdoor_and_natural']))
					photosets_info.update({photoset_id:photoset_info})
					with open(classification_file, 'wb') as fp:
						pickle.dump(photosets_info, fp)
					#print("image path: " + img_path + ", outdoor: " + str(outdoor) + ", natural: " + str(natural))
					#img = cv2.imread(img_path)
					#cv2.imshow('image',img)
					#cv2.waitKey(500)
	#cv2.destroyAllWindows()

def showDatasetImagesLargerThanThreshold(found_photos_file, outdoor_natural_threshold = 0.8, max_img=10):
	dataset_path = os.path.dirname(found_photos_file)
	showcase_path = os.path.join(dataset_path, "showcase")
	if not os.path.exists(showcase_path):
		os.makedirs(showcase_path)

	cls_file_name = getClassificationFileName(found_photos_file)
	cls_data = ""
	with open(cls_file_name, 'rb') as cfp:
		cls_data = pickle.load(cfp)

	outdoor_natural = np.array([])
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		for user_id in data['found_users']:
			photosets = data['found_users'][user_id]
			for photoset_id in photosets:
				if photoset_id in cls_data:
					images_info = cls_data[photoset_id]["images_info"]
					on = cls_data[photoset_id]["outdoor_and_natural"]
					if (on > outdoor_natural_threshold):
						#show this photoset
						image_dir = os.path.join(dataset_path, "images", user_id, photoset_id)
						images = []
						cnt = 0
						#print(image_dir)
						for filename in os.listdir(image_dir):
							img_path = os.path.join(image_dir, filename)
							if (images_info["outdoor"] and images_info["natural"]):
								images.append(mpimg.imread(img_path))
								cnt += 1
							if cnt > max_img:
								break
						#show images
						if (cnt > 0):
							fig = show_images(images, int(np.sqrt(cnt)), "Outdoor and natural images larger than threshold: " + str(outdoor_natural_threshold))
							figname = os.path.join(showcase_path, "positive_" + user_id + ":" + photoset_id + ".pdf")
							fig.savefig(figname, dpi = 500)   # save the figure to file
							print(figname)
							plt.close(fig)
					else:
						print("Value lower than threshold: " + str(on))
				else:
					print("Photoset id is not in classified data.")

def showDatasetImagesLowerThanThreshold(found_photos_file, outdoor_natural_threshold = 0.3, max_img=10):
	dataset_path = os.path.dirname(found_photos_file)
	showcase_path = os.path.join(dataset_path, "showcase")
	if not os.path.exists(showcase_path):
		os.makedirs(showcase_path)
	cls_file_name = getClassificationFileName(found_photos_file)
	cls_data = ""
	with open(cls_file_name, 'rb') as cfp:
		cls_data = pickle.load(cfp)

	outdoor_natural = np.array([])
	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		for user_id in data['found_users']:
			photosets = data['found_users'][user_id]
			for photoset_id in photosets:
				if photoset_id in cls_data:
					images_info = cls_data[photoset_id]["images_info"]
					on = cls_data[photoset_id]["outdoor_and_natural"]
					if (on < outdoor_natural_threshold):
						#show this photoset
						image_dir = os.path.join(dataset_path, "images", user_id, photoset_id)
						images = []
						cnt = 0
						print(image_dir)
						for filename in os.listdir(image_dir):
							img_path = os.path.join(image_dir, filename)
							#if (images_info["outdoor"] and images_info["natural"]):
							images.append(mpimg.imread(img_path))
							cnt += 1
							if cnt > max_img:
								break
						#show images
						if (cnt > 0):
							fig = show_images(images, int(np.sqrt(cnt)), "Outdoor and natural images lower than threshold: " + str(outdoor_natural_threshold))
							figname = os.path.join(showcase_path, "negative_" + user_id + ":" + photoset_id + ".pdf")
							fig.savefig(figname, dpi = 500)   # save the figure to file
							print(figname)
							plt.close(fig)
						else:
							print(image_dir + " - contains 0 images")
					else:
						print("Value greater than threshold: " + str(on))
				else:
					print("Photoset id is not in classified data.")

def downloadExif(photo_id, found_photos_file):
	#exif_data_file = getExifDataFileName(found_photos_file)
	#exif_data = {}
	#if os.path.isfile(exif_data_file):
	#	f = open(exif_data_file, 'rb')
	#	exif_data = pickle.load(f)
	#	f.close()
	#if photo_id in exif_data:
        #	return exif_data[photo_id]
	#download and save
	exif = flickr.photos.getExif(photo_id = photo_id)
	#exif_data.update({photo_id:exif})
	#with open(exif_data_file, 'wb') as fp:
	#	pickle.dump(exif_data, fp)

	return exif

def putExifToPhoto(exif, photo_path):
	with exiftool.ExifTool() as et:
		updated_cnt = 0
		for tag in exif['photo']['exif']:
			tag_name = tag['tag']
			content = tag['raw']['_content']
			if tag_name == "FocalLength":
				content = content.split(" ")[0]
				#print(tag_name + ": " + content)
			try:
				res = et.execute(("-" + tag_name + "=" + content).encode(), photo_path.encode())
				res = res.decode()
				num_updated = res.rstrip().split(" ")[0]
				if num_updated.isdigit():
					updated_cnt += int(num_updated)
				print("Updated " + str(updated_cnt) + " EXIF records of " + os.path.basename(photo_path))
			except Exception as exc:
				print("Unable to write exif to photo 1: ", exc)


def writeCommandExiftool(command, path):
	updated_cnt = 0
	with exiftool.ExifTool() as et:
		try:
			res = et.execute(command.encode(), path.encode())
			res = res.decode()
			num_updated = res.rstrip().split(" ")[0]
			if num_updated.isdigit():
				updated_cnt += int(num_updated)
		except Exception as exc:
			print("Unable to write exif to photo 2: ", exc)

	return updated_cnt



def putLocationToPhoto(loc_photoset, photo_id, photo_path):
	if photo_id in loc_photoset:
		location = loc_photoset[photo_id]["photo"]["location"]
		with exiftool.ExifTool() as et:
			updated_cnt = 0
			lat = float(location["latitude"])
			lon = float(location["longitude"])
			lat_ref = "N"
			if (lat < 0):
				lat_ref = "S"
			lon_ref = "E"
			if (lon < 0):
				lon_ref = "W"
			updated_cnt += writeCommandExiftool("-GPSLatitude=" + str(lat), photo_path)
			updated_cnt += writeCommandExiftool("-GPSLatitudeRef=" + lat_ref, photo_path)
			updated_cnt += writeCommandExiftool("-GPSLongitude=" + str(lon), photo_path)
			updated_cnt += writeCommandExiftool("-GPSLongitudeRef=" + lon_ref, photo_path)

			print("Location updated: " + str(updated_cnt) + " EXIF records of " + os.path.basename(photo_path))
	else:
		print("No photo in location data for photo id: " + photo_id)

def putInformationToPhotosInSingleDataset(found_photos_file):
	output_location_file = getLocationFileName(found_photos_file)
	if not os.path.isfile(output_location_file):
		print("Location data are not downloaded. Download the location data first.")
		return

	location_data = ""
	with open(output_location_file, 'rb') as fp:
		location_data = pickle.load(fp)

	dataset_path = os.path.dirname(found_photos_file)
	images_done = []
	images_done_path = os.path.join(dataset_path, "images_done.txt")
	if (os.path.exists(images_done_path)):
		with open(os.path.join(dataset_path, "images_done.txt")) as fp:
			images_done = [line.strip() for line in fp]
	image_dir = os.path.join(dataset_path, "images")
	existing_images = os.listdir(image_dir)
	if os.path.isdir(image_dir):
		#iterate through photos
		for filename in existing_images:
			if filename in images_done:
				print("Skipping: " + filename)
				continue
			img_path = os.path.join(image_dir, filename)
			photo_id = filename.split("_")[0]
			try:
				try:
					try:
						exif = downloadExif(photo_id, found_photos_file)
						putExifToPhoto(exif, img_path)
					except requests.exceptions.ConnectionError as ce:
						print("Connection error: ", ce, file = sys.stderr)
				except flickrapi.exceptions.FlickrError as fe:
					print("Flickr error: ", fe, file = sys.stderr)

				putLocationToPhoto(location_data, photo_id, img_path)
			except UnicodeEncodeError as uee:
				print("Unable to write exif data: ", uee, file = sys.stderr)
def putInformationToPhotosInPhotoset(found_photos_file, user_id, photoset_id):
	""" Location data for the given photoset must be already downloaded
	using downloadLocationData(). Iterates through all photos in the photoset,
	for each photo the exif is downloaded from Flickr (if not found on disk),
	and put with location data into the photo EXIF using ExifTool."""

	#test whether location data are downloaded
	output_location_file = getLocationFileName(found_photos_file)
	if not os.path.isfile(output_location_file):
		print("Location data are not downloaded. Download the location data first.")
		return

	location_data = ""
	with open(output_location_file, 'rb') as fp:
		location_data = pickle.load(fp)

	#num_photos_gps = len(location_data[photoset_id])
	#print("Number of photos with GPS: " + str(num_photos_gps))

	dataset_path = os.path.dirname(found_photos_file)

	image_dir = os.path.join(dataset_path, "images", user_id, photoset_id)
	if os.path.isdir(image_dir):
		#iterate through photos
		for filename in os.listdir(image_dir):
			img_path = os.path.join(image_dir, filename)
			photo_id = filename.split("_")[0]
			try:
				exif = downloadExif(photo_id, found_photos_file)
				putExifToPhoto(exif, img_path)
			except flickrapi.exceptions.FlickrError as fe:
				print("Flickr error: ", fe, file = sys.stderr)

			loc_photoset = location_data[photoset_id]
			putLocationToPhoto(loc_photoset, photo_id, img_path)

def purge(dir, pattern):
	for f in os.listdir(dir):
		if re.search(pattern, f):
			os.remove(os.path.join(dir, f))

def copyPhotosToReconstructionDir(found_photos_file, user_id, photoset_id):
	dataset_path = os.path.dirname(found_photos_file)
	image_dir = os.path.join(dataset_path, "images", user_id, photoset_id)
	reconstruction_dir = os.path.join(dataset_path, "reconstruction", photoset_id, 'perspective')

	if not os.path.exists(reconstruction_dir):
		os.makedirs(reconstruction_dir)

	#purge backups of images generated by exiftool
	purge(image_dir, ".*_original")

	#copy all photos to reconstruction dir
	src_files = os.listdir(image_dir)
	for file_name in src_files:
		full_file_name = os.path.join(image_dir, file_name)
		if (os.path.isfile(full_file_name)):
			shutil.copy(full_file_name, reconstruction_dir)

def prepareDataForReconstruction(found_photos_file):
	#classification data
	cls_file_name = getClassificationFileName(found_photos_file)
	if not os.path.isfile(cls_file_name):
		print("Classification data are not available. Classify the data first.")
		return
	cls_data = ""
	with open(cls_file_name, 'rb') as cfp:
		cls_data = pickle.load(cfp)

	#location data
	output_location_file = getLocationFileName(found_photos_file)
	if not os.path.isfile(output_location_file):
		print("Location data are not downloaded. Download the location data first.")
		return

	location_data = ""
	with open(output_location_file, 'rb') as fp:
		location_data = pickle.load(fp)

	with open(found_photos_file, 'rb') as fp:
		data = pickle.load(fp)
		for user_id in data['found_users']:
			photosets = data['found_users'][user_id]
			for photoset_id in photosets:

				#if there is no location information, don't use this album
				if not photoset_id in location_data:
					continue
				#if there is no classification information, don't use this album
				if not photoset_id in cls_data:
					continue

				#if there is less than 60 of photos, don't use this album
				num_photos = int(data['photoset_data'][photoset_id]['photos'])
				if num_photos < 60:
					continue

				num_photos_gps = len(location_data[photoset_id])
				#if there are not at least 50% of photos with gps, don't use this album
				if (float(num_photos_gps)/float(num_photos) < 0.5):
					continue

				on = cls_data[photoset_id]["outdoor_and_natural"]
				#if there are less than 60% of photos that are outdoor and
				#natural, don't use this album
				if on < 0.6:
					continue

				#all conditions passed, use this album
				#put all information needed into the photos - exif and gps
				print("Preparing album - user_id: " + user_id + ", photoset id: " + photoset_id)
				#putInformationToPhotosInPhotoset(found_photos_file, user_id, photoset_id)

				copyPhotosToReconstructionDir(found_photos_file, user_id, photoset_id)
				print("Prepared album - user_id: " + user_id + ", photoset id: " + photoset_id)

def buildArgumentParser():
	parser = argparse.ArgumentParser()
	parser.add_argument("output_directory", help="The directory to store the downloaded images and metadata.")
	parser.add_argument("latitude", help="Latitude of the center of the circlular area to search for the photos", type=float)
	parser.add_argument("longitude", help="Longitude of the center of the circular area to search for the photos", type=float)
	parser.add_argument("radius", help="Radius in km of the circle defining the area to search for the photos, must be lower than 32 km", type=float)
	parser.add_argument("-p", "--photos_count", help="How many photographs shall be downloaded. Negative number means to download as many photos as are available. Default=-1.", type=int, default=-1)
	parser.add_argument("-s", "--skip_photos", help="Skip downloading photographs -- if photographs are already downloded. The process continues with downloading GPS and EXIF for each photograph.", action="store_true")
	parser.add_argument("-d", "--dry-run", help="Just query the count of images in the specified area, don't download anything.", action="store_true")

	return parser

def main():

	parser = buildArgumentParser()
	args = parser.parse_args()

	output_dir = args.output_directory
	found_photos_file = os.path.join(output_dir, os.path.basename(output_dir) + ".p")

	if args.radius > 32:
		raise ValueError("Radius can be at most 32 km due to Flickr restrictions.")

	if (not args.skip_photos):
		# skip downloading photographs as they are already downloaded
		searchAndDownloadPhotosInGeoCircleWithCount(args.latitude, args.longitude, args.radius, args.photos_count, found_photos_file)

	downloadLocationDataForSingleDataset(found_photos_file)
	putInformationToPhotosInSingleDataset(found_photos_file)


if __name__ == "__main__":
	main()
