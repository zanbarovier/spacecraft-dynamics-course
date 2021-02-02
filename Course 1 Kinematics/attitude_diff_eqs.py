import numpy as np
import attitude_coords_library as ac

def eul_ang_eqset(t,x):
    """ Sets the differential equations for euler angles"""
    y = x[0] # yaw
    p = x[1] # pitch
    r = x[2] # roll
    w = (20 * np.pi / 180) * np.array([[np.sin(0.1 * t)],[0.01],[np.cos(0.1 * t)]])
    dcm = np.array([[-np.sin(p), 0, 1],[np.sin(r) * np.cos(p), np.cos(r), 0],[np.cos(r) * np.cos(p), -np.sin(r), 0]])
    dcm_inv = np.linalg.inv(dcm)
    dyneq = np.matmul(dcm_inv,w)
    dyneq = np.transpose(dyneq)

    return dyneq

def quat_eqset(t,x):
    """Sets the differential equations for quaternions"""
    b0,b1,b2,b3 = x[0], x[1], x[2], x[3]
    w = (20*np.pi/180)*np.array([np.sin(0.1*t),0.01,np.cos((0.1*t))])
    wquat = np.array([0,w[0],w[1],w[2]])
    wquat = np.transpose(wquat)
    bmat = np.array([[b0,-b1,-b2,-b3],[ b1, b0, -b3, b2],[ b2, b3, b0, -b1],[ b3, -b2, b1, b0]])
    qdot = 0.5*np.matmul(bmat,wquat)
    qdot = np.transpose(qdot)

    return qdot

def crp_eqset(t,x):
    """ Sets the differential equations for classical rodrigues parameters"""
    q = x
    q_tilde = ac.tilde_matrix(q)
    w = (3*np.pi/180)*np.array([np.sin(0.1*t),0.01,np.cos(0.1*t)])
    w = np.transpose(w)
    bmat = 0.5*(np.identity(3)+q_tilde+np.outer(q,q))
    qdot = np.matmul(bmat,w)
    qdot = np.transpose(qdot)

    return qdot

def mrp_eqset(t,x):
    """ Sets the differential equations for modified rodrigues parameters"""
    sig = x
    sig_tilde = ac.tilde_matrix(sig)
    w = (20*np.pi/180)*np.array([np.sin(0.1*t),0.01,np.cos(0.1*t)])
    w = np.transpose(w)
    bmat = 0.25*((1-np.dot(sig,sig))*np.identity(3)+2*sig_tilde+2*np.outer(sig,sig))
    sigdot = np.matmul(bmat,w)
    sigdot = np.transpose(sigdot)

    return sigdot