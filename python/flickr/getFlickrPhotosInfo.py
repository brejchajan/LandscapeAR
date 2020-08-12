# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-04T11:51:50+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-12T11:41:50+02:00
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




import argparse as ap
import flickrapi
from FlickrPrefs import FlickrPrefs
import pickle
from tqdm import tqdm

prefs = FlickrPrefs()
flickr = prefs.flickr
per_page = 500

def buildArgumentParser():
    parser = ap.ArgumentParser()
    parser.add_argument("--names", help="Specify the file containing one\
                        flickr photo id per line. Photo information will \
                        be downloaded for each photo id and stored in local\
                        database file.")
    parser.add_argument("--output-file", help="Name of the output database \
                        file. Default database.pickle",
                        default="database.pickle")
    parser.add_argument("--database", help="Name of the database to use. \
                        If this option is used, the tool can be used to \
                        query the database using various options.")
    parser.add_argument("--license", action='store_true', help="Print out the license of each \
                        file. Can be used only with --database option.")
    return parser


def getInfoForPhotos(photo_ids, output_file):
    all_info = {}
    idx = 0
    for photo_key in tqdm(photo_ids):
        try:
            photo_key_parts = photo_key.split('_')
            photo_id = photo_key_parts[0]
            photo_secret = photo_key_parts[1]
            info = flickr.photos.getInfo(photo_id=photo_id, secret=photo_secret)
            all_info.update({photo_key:info})
            if idx % 10 == 0:
                print("Backing up database to: ", output_file)
                with open(output_file, 'wb') as of:
                    pickle.dump(all_info, of)
                idx = 0
            idx += 1
        except flickrapi.exceptions.FlickrError as fe:
            print("Unable to process photo id: " + photo_id
                  + ", reason: ", fe)
    return all_info


def printLicenses(db, photo_ids):
    licenses_info = flickr.photos.licenses.getInfo()
    numlic = len(licenses_info['licenses']['license'])
    license_ids = {}
    for idx in range(0, numlic):
        lic_id = int(licenses_info['licenses']['license'][idx]['id'])
        lic_name = licenses_info['licenses']['license'][idx]['name']
        license_ids.update({lic_id:lic_name})
    #print(license_ids[1])
    for photo_key in tqdm(photo_ids):
        if photo_key in db:
            license_id = int(db[photo_key]['photo']['license'])
            license = license_ids[license_id]
            print(photo_key, license_id, license)
        else:
            print(photo_key, -1, "not found")



def main():
    parser = buildArgumentParser()
    args = parser.parse_args()

    if args.database:
        with open(args.database, 'rb') as df:
            db = pickle.load(df)

    if args.names:
        with open(args.names, 'r') as f:
            photo_ids = [line.strip() for line in f.readlines()]

        if not args.database:
            info = getInfoForPhotos(photo_ids, args.output_file)
            with open(args.output_file, 'wb') as of:
                pickle.dump(info, of)
        else:
            if args.license:
                printLicenses(db, photo_ids)


if __name__ == "__main__":
    main()
