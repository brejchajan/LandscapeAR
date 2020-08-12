# @Author: Jan Brejcha <janbrejcha>
# @Email:  brejcha@adobe.com, ibrejcha@fit.vutbr.cz, brejchaja@gmail.com
# @Project: ImmersiveTripReports 2017-2018, LandscapeAR 2018
# AdobePatentID="P7840-US"
# @License: Copyright 2020 CPhoto@FIT, Brno University of Technology,
# Faculty of Information Technology,
# Božetěchova 2, 612 00, Brno, Czech Republic
#
# Redistribution and use in source code form, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
#
# 2. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# 3. Redistributions must be pursued only for non-commercial research
#    collaboration and demonstration purposes.
#
# 4. Where separate files retain their original licence terms
#    (e.g. MPL 2.0, Apache licence), these licence terms are announced, prevail
#    these terms and must be complied.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.


import flickrapi

class FlickrPrefs:

	def __init__(self):
		self.flickr_key = '4066e1e745a514da1ff09bfdfe064c4c'
		self.flickr_secret = '2fad5d7599a12f4e'
		self.flickr = flickrapi.FlickrAPI(self.flickr_key, self.flickr_secret, format='parsed-json')