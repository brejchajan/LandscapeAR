# @Author: Jan Brejcha <ibrejcha>
# @Date:   2020-08-14T10:43:41+02:00
# @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
# @Project: LandscapeAR
# @Last modified by:   ibrejcha
# @Last modified time: 2020-08-14T10:45:18+02:00
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

import sys
from pose_estimation.eulerZYZ import rot_matrix


def angles_to_xml(a, b, g):
    m = rot_matrix(float(a), float(b), float(g))
    print("<!DOCTYPE ObjectXML>")
    print("<Object>")
    print(" <C0 x0=\""+str(m[0, 0])+"\" x1=\""+str(m[0, 1])+"\" x2=\""+str(m[0, 2])+"\"/>")
    print(" <C1 x0=\""+str(m[1, 0])+"\" x1=\""+str(m[1, 1])+"\" x2=\""+str(m[1, 2])+"\"/>")
    print(" <C2 x0=\""+str(m[2, 0])+"\" x1=\""+str(m[2, 1])+"\" x2=\""+str(m[2, 2])+"\"/>")
    print("</Object>")


if __name__ == "__main__":
    """
    Simple utility to convert rotation angles from GeoPose3K to rotationC2G.xml
    usable by itr for rendering.
    """
    angles_to_xml(sys.argv[1], sys.argv[2], sys.argv[3])
