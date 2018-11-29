'''Transform one mesh into another'''
import sys
import os
sys.path.insert(0, "libigl/python")
import pyigl as igl
import numpy as np
from iglhelpers import *
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
import scipy
from scipy.optimize import fmin_l_bfgs_b

Vref = None
Sref = None
Fref = None
Smov = None
Fmov = None
vecs = None
k = None
UxLUx = None
Ux = None
UyLUy = None
Uy = None
UzLUz = None
Uz = None
L = None
it = None

def project(smov):
    '''
    Project vertices of moving sphere into reference sphere and
    then project the vertices of the native moving vertices into the native
    reference mesh
    '''
    # smov is eigen, dimensions nverts * 3
    global Vref, Sref, Fref
    # compute closest points of smv in Sref
    sqrD = igl.eigen.MatrixXd()
    I = igl.eigen.MatrixXi()
    C = igl.eigen.MatrixXd()
    igl.point_mesh_squared_distance(smov, Sref, Fref, sqrD, I, C)
    # get barycentric coordinates
    Vx = igl.eigen.MatrixXd()
    Vy = igl.eigen.MatrixXd()
    Vz = igl.eigen.MatrixXd()
    xyz = p2e(np.array([0, 1, 2]))
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.slice(Fref, I, xyz, F)
    igl.slice(Sref, F.col(0), xyz, Vx)
    igl.slice(Sref, F.col(1), xyz, Vy)
    igl.slice(Sref, F.col(2), xyz, Vz)
    B = igl.eigen.MatrixXd()
    igl.barycentric_coordinates(C, Vx, Vy, Vz, B)
    # get coordinates in Vref space
    igl.slice(Vref, F.col(0), xyz, Vx)
    igl.slice(Vref, F.col(1), xyz, Vy)
    igl.slice(Vref, F.col(2), xyz, Vz)
    V = igl.eigen.MatrixXd(smov)
    for i in range(smov.rows()):
        V.setRow(i, Vx.row(i)*B[i, 0] + Vy.row(i)*B[i, 1] + Vz.row(i)*B[i, 2])

    return V # V is eigen

def sphere(coords):
    '''
    Convert eigenvector coordinates back into a sphere
    '''
    global Smov, vecs, k
    a = coords[0:k]
    b = coords[k:2*k]
    c = coords[2*k:3*k]
    delt = np.concatenate(([vecs.dot(a)], [vecs.dot(b)], [vecs.dot(c)])).T
    sph = e2p(Smov) + delt
    sph = p2e(sph).rowwiseNormalized()

    return sph # eigen

def flip_energy(smov):
    '''
    Compute energy associated to morph triangles deviating from
    the sphere's normal as an angle. The angle is 0 for perfect
    alignment, and pi for inverted triangles.
    '''
    global Fmov
    morph_normals = igl.eigen.MatrixXd()
    igl.per_vertex_normals(smov, Fmov, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,morph_normals)
    sphere_normals = smov.rowwiseNormalized()
    dot = np.einsum('ij,ij->i', e2p(morph_normals), e2p(sphere_normals))
    ang = np.power(np.arccos(dot), 2)
    E = np.sum(ang)

    return E

def energy(coords):
    '''
    Compute the deformation energy
    '''
    global UxLUx, Ux, UyLUy, Uy, UzLUz, Uz, L
    sph = sphere(coords) # sph is eigen
    flipE = 0 # flip_energy(sph) # flipE is a scalar
    VmovProj = project(sph) # VmovProj is eigen
    X = VmovProj.col(0)
    Y = VmovProj.col(1)
    Z = VmovProj.col(2)
    Ex = UxLUx + X.transpose()*L*X - 2 * Ux.transpose()*L*X
    Ey = UyLUy + Y.transpose()*L*Y - 2 * Uy.transpose()*L*Y
    Ez = UzLUz + Z.transpose()*L*Z - 2 * Uz.transpose()*L*Z
    E = e2p(Ex + Ey + Ez)[0, 0] + 0.1 * flipE

    return E # result is a single scalar

def iteration(x):
    '''
    Callback function that is called after every iteration of the minimisation algorithm
    '''
    global it
    it = it + 1
    E = energy(x)
    print("iter:", it, ", energy:", E)

def undeform(path_ref, path_mov, path_morph, outdir):
    global Vref, Sref, Fref, Smov, Fmov
    global vecs, k
    global UxLUx, Ux, UyLUy, Uy, UzLUz, Uz, L
    global it
    '''
    Specific paths
    '''
    path_vref = path_ref + "/surf.ply"
    path_vmov = path_mov + "/surf.ply"
    path_sref = path_ref + "/surf.sphere.ply"
    path_smov = path_mov + "/surf.sphere.ply"
    path_rotref = path_ref + "/rotation.txt"
    path_rotmov = path_mov + "/rotation.txt"

    '''
    Load meshes
    '''
    Vref = igl.eigen.MatrixXd()
    Vmov = igl.eigen.MatrixXd()
    Sref = igl.eigen.MatrixXd()
    Smov = igl.eigen.MatrixXd()
    Smor = igl.eigen.MatrixXd()
    Fref = igl.eigen.MatrixXi()
    Fmov = igl.eigen.MatrixXi()
    TMP1 = igl.eigen.MatrixXd()
    TMP2 = igl.eigen.MatrixXd()

    # load meshes
    igl.readPLY(path_vref, Vref, Fref, TMP1, TMP2)
    igl.readPLY(path_vmov, Vmov, Fmov, TMP1, TMP2)
    igl.readPLY(path_sref, Sref, Fref, TMP1, TMP2)
    igl.readPLY(path_smov, Smov, Fmov, TMP1, TMP2)
    igl.readPLY(path_morph, Smor, Fmov, TMP1, TMP2)

    # load and apply rotations
    rot = np.loadtxt(path_rotref)
    rot = p2e(rot).leftCols(3).topRows(3)
    Sref = (rot.transpose()*Sref.transpose()).transpose()

    rot = np.loadtxt(path_rotmov)
    rot = p2e(rot).leftCols(3).topRows(3)
    Smov = (rot.transpose()*Smov.transpose()).transpose()

    '''
    Compute uniform Laplacian matrix
    '''
    A = igl.eigen.SparseMatrixi()
    igl.adjacency_matrix(Fmov, A)
    pA = e2p(A).astype('float64')
    pL = scipy.sparse.csgraph.laplacian(pA).astype(float).tocsc()
    L = p2e(pL)

    '''
    Eigenpairs of the uniform Laplacian
    '''
    k = 25
    print("Using " + str(k) + " eigenvectors");
    vals, vecs = eigsh(pL, k, sigma=0, which='LM')
    evecs = p2e(vecs)

    '''
    Initial deformation field guess
    '''
    delta = Smor - Smov
    a = delta.col(0).transpose()*evecs # 1 row, k columns
    b = delta.col(1).transpose()*evecs
    c = delta.col(2).transpose()*evecs
    abc = np.concatenate((a, b, c), axis=1).T[:, 0]

    '''
    Constant part of the deformation energy function
    '''
    Ux = Vmov.col(0)
    Uy = Vmov.col(1)
    Uz = Vmov.col(2)
    UxLUx = Ux.transpose()*L*Ux
    UyLUy = Uy.transpose()*L*Uy
    UzLUz = Uz.transpose()*L*Uz

    '''
    Initialise AABB tree for finding closest vertices
    '''
    tree = igl.AABB()
    tree.init(Sref, Fref)

    '''
    Call the minimisation algorithm
    '''
    it = 0
    x, f, d = fmin_l_bfgs_b(energy, x0=abc, approx_grad=True, callback=iteration)

    '''
    Save the resulting morph sphere and the resulting projected mesh
    '''
    sph = sphere(x)
    igl.writePLY(outdir + "/surf.sphere.ply", sphere(x), Fmov)
    igl.writePLY(outdir + "/surf.ply", project(sph), Fmov)

def main(argv):
    '''
    Read arguments
    '''
    path_ref = sys.argv[1]
    path_mov = sys.argv[2]
    path_morph = sys.argv[3]
    outdir = sys.argv[4]

    undeform(path_ref, path_mov, path_morph, outdir)

if __name__ == "__main__":
    main(sys.argv)
