# @Author: Jan Brejcha <janbrejcha>
# @Email:  brejcha@adobe.com, ibrejcha@fit.vutbr.cz, brejchaja@gmail.com
# @Project: ImmersiveTripReports 2017-2018, LandscapeAR 2018
# AdobePatentID="P7840-US"

import flickrapi

class FlickrPrefs:

	def __init__(self):
		self.flickr_key = '4066e1e745a514da1ff09bfdfe064c4c'
		self.flickr_secret = '2fad5d7599a12f4e'
		self.flickr = flickrapi.FlickrAPI(self.flickr_key, self.flickr_secret, format='parsed-json')
