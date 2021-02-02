import numpy as np
import attitude_coords_library as cl
import sympy as sym

def triad(v1B,v2B,v1N,v2N):
    """ Implements the triad method for attitude estimation, assuming
    v1 is the higher confidence measurement. B is body frame, N is inertial frame"""
    # ensure all vectors are normalized
    v1B = v1B/np.linalg.norm(v1B)
    v1N = v1N/np.linalg.norm(v1N)
    v2B = v2B/np.linalg.norm(v2B)
    v2N = v2N/np.linalg.norm(v2N)

    # t-frame in body coordinates
    t1B = v1B
    t2B = np.cross(v1B,v2B)/np.linalg.norm(np.cross(v1B,v2B))
    t3B = np.cross(t1B,t2B)
    BbarT = np.array([t1B,t2B,t3B])
    BbarT = np.transpose(BbarT)

    # t-frame in inertial coordinates
    t1N = v1N
    t2N = np.cross(v1N,v2N)/np.linalg.norm(np.cross(v1N,v2N))
    t3N = np.cross(t1N,t2N)
    NT = np.array([t1N,t2N,t3N])
    NT = np.transpose(NT)

    BbarN = np.matmul(BbarT,np.transpose(NT))

    return BbarN

def devenport(vec_B,vec_N,w):
    """ Implements devenport's method for attitude estimation, assuming B is body frame, N is inertial frame
    - vec_B is a list of  the  vectors in body frame coordinates, vec_N is a list of the vectors in inertial coordinates
    - w is a list of weights, in the order [w1,w2....wn]
    - Refer to the sympy documentation for more about how it works: https://www.sympy.org/en/index.html"""
    B = np.zeros([3,3]) # initialize B matrix
    # ensure all vectors are normalized
    for i in range(len(vec_B)):
        vec_B[i] = vec_B[i]/np.linalg.norm(vec_B[i])
        vec_N[i] = vec_N[i]/np.linalg.norm(vec_N[i])
        B += w[i]*np.outer(vec_B[i],vec_N[i])

    S = B + np.transpose(B)
    sig = np.trace(B)
    Z = np.array([B[1,2]-B[2,1],B[2,0]-B[0,2],B[0,1]-B[1,0]])
    Z = np.transpose(Z)
    col1 = np.array([[sig], [Z[0]], [Z[1]], [Z[2]]])
    col234 = np.concatenate([[np.transpose(Z)], S - sig * np.identity(3)])
    K = np.hstack((col1, col234))
    p,d = np.linalg.eig(K) # gets eigenvectors(d) and eigenvalues(p)
    max_lam = np.argmax(p)
    beta = d[:,max_lam]
    beta = beta/np.linalg.norm(beta)
    if beta[0]<1:
        beta = -beta

    dcm = cl.quat2dcm(beta)

    return dcm

def quest(vec_B,vec_N,w):
    """ Implements the quest method for attitude estimation, assuming B is body frame, N is inertial frame
    - vec_B is a list of  the  vectors in body frame coordinates, vec_N is a list of the vectors in inertial coordinates
    - w is a list of weights, in the order [w1,w2....wn]"""
    B = np.zeros([3, 3])  # initialize B matrix
    # ensure all vectors are normalized
    for i in range(len(vec_B)):
        vec_B[i] = vec_B[i] / np.linalg.norm(vec_B[i])
        vec_N[i] = vec_N[i] / np.linalg.norm(vec_N[i])
        B += w[i] * np.outer(vec_B[i], vec_N[i])

    S = B + np.transpose(B)
    sig = np.trace(B)
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    Z = np.transpose(Z)
    col1 = np.array([[sig], [Z[0]], [Z[1]], [Z[2]]])
    col234 = np.concatenate([[np.transpose(Z)], S - sig * np.identity(3)])
    K = np.hstack((col1, col234))
    K = sym.Matrix(K) # convert to sympy matrix
    x = sym.symbols('x')
    K_lam = K - x * sym.eye(4)
    f = K_lam.det('berkowitz')
    df = sym.diff(f)

    lam_guess = sum(w)
    lam1 = lam_guess-(f.subs(x,lam_guess)/df.subs(x,lam_guess)) # newton's method

    intermed_mat = np.array(((lam1+sig)*np.identity(3)-S),dtype='float')
    q = np.matmul(np.linalg.inv(intermed_mat),Z)

    dcm = cl.crp2dcm(q)

    return dcm


def olae(v1B,v2B,v1N,v2N,w):
    """ Implements the olae method for attitude estimation, assuming B is body frame, N is inertial frame
    - w is a list of weights, in the order [w1,w2....wn]"""
    # ensure all vectors are normalized
    v1B = v1B/np.linalg.norm(v1B)
    v1N = v1N/np.linalg.norm(v1N)
    v2B = v2B/np.linalg.norm(v2B)
    v2N = v2N/np.linalg.norm(v2N)

    s1 = v1B + v1N
    s2 = v2B + v2N
    d1 = v1B - v1N
    d2 = v2B - v2N

    s1_tilde = cl.tilde_matrix(s1)
    s2_tilde = cl.tilde_matrix(s2)

    S = np.vstack((s1_tilde, s2_tilde))
    d = np.hstack((d1,d2))

    W = np.vstack((np.hstack((w[0]*np.identity(3),np.zeros([3,3]))),np.hstack((np.zeros([3,3]),w[1]*np.identity(3)))))
    intermed_mat = np.matmul(np.transpose(S),np.matmul(W,S))
    inv_mat = np.linalg.inv(intermed_mat)
    q = np.matmul(W,d)
    q = np.matmul(np.transpose(S),q)
    q = np.matmul(inv_mat,q)
    dcm = cl.crp2dcm(q)
    return dcm






