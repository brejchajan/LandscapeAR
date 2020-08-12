import numpy as np
import pyp4pf

# find rigid transformation (rotation translation) from p1->p2 given
# 3 or more points in the 3D space
#
# based on : K.Arun, T.Huangs, D.Blostein. Least-squares
#            fitting of two 3D point sets IEEE PAMI 1987
#
# rewritten by Jan Brejcha (Jun 2020) from the
# original Matlab code by Martin Bujnak, nov2007
#
#function [R t] = GetRigidTransform(p1, p2, bLeftHandSystem)
# p1 - 3xN matrix with reference 3D points
# p2 - 3xN matrix with target 3D points
# bLeftHandSystem - camera coordinate system


def getRigidTransform2(p1, p2, bLeftHandSystem):

    N = p1.shape[1]
    # shift centers of gravity to the origin
    p1mean = (np.sum(p1, axis=1) / N).reshape(-1, 1)
    p2mean = (np.sum(p2, axis=1) / N).reshape(-1, 1)
    p1 = p1 - p1mean
    p2 = p2 - p2mean

    # normalize to unit size
    rep1 = (1.0 / np.sqrt(np.sum(np.power(p1, 2), axis=0))).reshape(1, -1)
    u1 = p1 * rep1
    rep2 = (1.0 / np.sqrt(np.sum(np.power(p2, 2), axis=0))).reshape(1, -1)
    u2 = p2 * rep2

    # calc rotation
    C = np.dot(u2, u1.transpose())
    U, S, Vt = np.linalg.svd(C)

    # fit to rotation space
    S[0] = np.sign(S[0])
    S[1] = np.sign(S[1])
    if bLeftHandSystem:
        S[2] = -np.sign(np.linalg.det(np.dot(U, Vt)))
    else:
        S[2] = np.sign(np.linalg.det(np.dot(U, Vt)))

    R = np.dot(np.dot(U, np.diag(S)), Vt)
    t = (np.dot(-R, p1mean) + p2mean).reshape(-1)

    return R, t


def p4pf(m2D, M3D):
    tol = 2.2204e-10
    # normalize 2D, 3D

    R = None
    t = None
    f = None

    # shift 3D data so that variance = sqrt(2), mean = 0
    mean3d = np.sum(M3D, axis=1) / 4
    M3D = M3D - mean3d.transpose().reshape(-1, 1)

    # variance (isotropic)
    var = np.sum(np.sqrt(np.sum(np.power(M3D, 2), axis=0))) / 4
    M3D = (1/var)*M3D

    # scale 2D data
    var2d = var2d = np.sum(np.sqrt(np.sum(np.power(m2D, 2), axis=0))) / 4
    m2D = (1/var2d)*m2D

    # caclulate quadratic distances between 3D points
    glab = np.sum(np.power(M3D[:, 0] - M3D[:, 1], 2))
    glac = np.sum(np.power(M3D[:, 0] - M3D[:, 2], 2))
    glad = np.sum(np.power(M3D[:, 0] - M3D[:, 3], 2))
    glbc = np.sum(np.power(M3D[:, 1] - M3D[:, 2], 2))
    glbd = np.sum(np.power(M3D[:, 1] - M3D[:, 3], 2))
    glcd = np.sum(np.power(M3D[:, 2] - M3D[:, 3], 2))

    if glab*glac*glad*glbc*glbd*glcd < tol:
        # initial solution degeneracy - invalid input
        return R, t, f

    gl_all = np.array([glab, glac, glad, glbc, glbd, glcd])
    # print(gl_all)
    # print(m2D[:, 0], m2D[:, 1], m2D[:, 2], m2D[:, 3])
    A = pyp4pf.p4pf(gl_all, m2D[:, 0], m2D[:, 1], m2D[:, 2], m2D[:, 3])
    # print(A)

    D, V = np.linalg.eig(A)

    sol = V[1:5, :] / (np.ones([4, 1]) * V[0, :])
    if np.sum(np.isnan(sol)) > 0:
        return R, t, f
    else:

        imagsol = np.imag(sol[3, :])
        I = np.array(np.where(np.logical_not(imagsol))).reshape(-1)
        fidx =np.array(np.nonzero(np.real(sol[3, I]).reshape(-1) > 0)).reshape(-1)
        f = np.sqrt(np.real(sol[3, I[fidx]]))
        zd = np.real(sol[0, I[fidx]]);
        zc = np.real(sol[1, I[fidx]]);
        zb = np.real(sol[2, I[fidx]]);

        # recover camera rotation and translation
        lcnt = len(f)
        p3dc = np.zeros((3, 4))
        d = np.zeros(6)
        if lcnt > 0:
            R = []
            t = []
            for idx in range(0, lcnt):
                # create p3d points in a camera coordinate system
                # (using depths)
                p3dc[:, 0] = 1 * np.hstack([m2D[:, 0], f[idx]])
                p3dc[:, 1] = zb[idx] * np.hstack([m2D[:, 1], f[idx]])
                p3dc[:, 2] = zc[idx] * np.hstack([m2D[:, 2], f[idx]])
                p3dc[:, 3] = zd[idx] * np.hstack([m2D[:, 3], f[idx]])

                # fix scale (recover 'za')
                d[0] = np.sqrt(glab / (np.sum(np.power(p3dc[:, 0] - p3dc[:, 1], 2), axis=0)))
                d[1] = np.sqrt(glac / (np.sum(np.power(p3dc[:, 0] - p3dc[:, 2], 2), axis=0)))
                d[2] = np.sqrt(glad / (np.sum(np.power(p3dc[:, 0] - p3dc[:, 3], 2), axis=0)))
                d[3] = np.sqrt(glbc / (np.sum(np.power(p3dc[:, 1] - p3dc[:, 2], 2), axis=0)))
                d[4] = np.sqrt(glbd / (np.sum(np.power(p3dc[:, 1] - p3dc[:, 3], 2), axis=0)))
                d[5] = np.sqrt(glcd / (np.sum(np.power(p3dc[:, 2] - p3dc[:, 3], 2), axis=0)))

                # all d(i) should be equal...
                gta = np.sum(d) / 6
                p3dc = gta * p3dc

                Rr, tt = getRigidTransform2(M3D, p3dc, False)
                R.append(Rr)
                ct = np.dot(var, tt) - np.dot(Rr, mean3d.transpose())
                t.append(ct)
                f[idx] = np.dot(var2d, f[idx])

    return R, t, f


if __name__ == "__main__":
    m2D = m2D = np.array([0.1101,  0.3978,  0.2875, -0.0502,
                          0.0388, -0.1730, -0.0579,  0.1930]).reshape(2, 4)
    M3D = np.array([-3.3364, -23.3555,  -13.1894,    6.4316,
                    0.6595,  -4.9038,    1.1710,    0.1458,
                    -8.4666,  -3.9988,   -3.0225,  -22.1609]) .reshape(3, 4)
    R, t, f = p4pf(m2D, M3D)
    print("R", R)
    print("t", t)
    print("f", f)
