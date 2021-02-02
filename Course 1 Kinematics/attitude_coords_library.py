import numpy as np

def tilde_matrix(vec):
    x1 = vec[0]
    x2 = vec[1]
    x3 = vec[2]
    tilde_mat = np.array([[0, -x3, x2],[x3, 0, -x1],[-x2, x1, 0]])
    return tilde_mat

def ea2dcm(eul_ang,seq):
    """Converts Euler angle set to DCM, with inputs:
        -eul_ang : vector representing euler angles (radians)
        -seq : sequence of euler angles (i.e. 321)
    """
    th1 = eul_ang[0]
    th2 = eul_ang[1]
    th3 = eul_ang[2]
    if seq == 321:
        dcm = np.array([[np.cos(th1)*np.cos(th2), np.cos(th2)*np.sin(th1), -np.sin(th2)],
    [np.sin(th1)*np.sin(th2)*np.cos(th1)-np.cos(th3)*np.sin(th1), np.sin(th3)*np.sin(th2)*np.sin(th1)+np.cos(th3)*np.cos(th1), np.sin(th3)*np.cos(th2)],
    [np.cos(th3)*np.sin(th2)*np.cos(th1)+np.sin(th3)*np.sin(th1),np.cos(th3)*np.sin(th2)*np.sin(th1)-np.sin(th3)*np.cos(th1), np.cos(th3)*np.cos(th2)]])

    elif seq == 313:
        dcm = np.array([[np.cos(th3) * np.cos(th1)-np.sin(th3)*np.cos(th2)*np.sin(th1), np.cos(th3)*np.sin(th1)+np.sin(th3)*np.cos(th2)*np.cos(th1), np.sin(th3)*np.sin(th2)],
                        [-np.sin(th3) * np.cos(th1) - np.cos(th3) * np.cos(th2) * np.sin(th1),
                         -np.sin(th3) * np.sin(th1) + np.cos(th3) * np.cos(th2) * np.cos(th1),
                         np.cos(th3) * np.sin(th2)],
                        [np.sin(th2) * np.sin(th1),
                         -np.sin(th2) * np.cos(th1),
                         np.cos(th2)]])
    return dcm


def dcm2ea(dcm,seq):
    """Converts DCM  to euler angles (in degrees), with inputs:
        -DCM : direction cosine matrix
        -seq : sequence of euler angles (i.e. 321)
    """
    rad2deg = (180/np.pi)
    if seq == 321:
        th1 = rad2deg*np.arctan2(dcm[0,1],dcm[0,0])
        th2 = -rad2deg*np.arcsin(dcm[0,2])
        th3 = rad2deg*np.arctan2(dcm[1,2],dcm[2,2])
    elif seq == 313:
        th1 = rad2deg*np.arctan2(dcm[2,0],-dcm[2,1])
        th2 = rad2deg*np.arccos(dcm[2,2])
        th3 = rad2deg*np.arctan2(dcm[0,2],dcm[1,2])



    eul_ang = np.array([[th1],[th2],[th3]])

    return eul_ang

def prv2dcm(phi,ehat):
    """ Converts a principle rotation vector to a dcm
    - phi : rotation angle (radians)
    - ehat : unit vector for rotation axis
    """
    sig = 1-np.cos(phi)
    e1,e2,e3 = ehat[0], ehat[1], ehat[2]

    dcm = np.array([[sig*e1**2+np.cos(phi),e1*e2*sig+e3*np.sin(phi), e1*e3*sig-e2*np.sin(phi)],
            [e2*e1*sig-e3*np.sin(phi),sig*e2**2+np.cos(phi),e2*e3*sig+e1*np.sin(phi)],
            [e3*e1*sig+e2*np.sin(phi),e3*e2*sig-e1*np.sin(phi)],sig*e3**2+np.cos(phi)])

    return dcm

def dcm2prv(dcm):
    """ Converts  dcm to a principle rotation vector"""
    phi = np.arccos(0.5*(dcm[0,0]+dcm[1,1]+dcm[2,2]-1))
    ehat = (0.5/np.sin(phi))*np.array([dcm[1,2]-dcm[2,1],dcm[2,0]-dcm[0,2],dcm[0,1]-dcm[1,0]])
    return phi,ehat

def prv_add(prv1,prv2):
    """ Adds two prvs together
    prv1,prv2 of format [phi, [ehat]]"""
    phi1 = prv1[0]
    ehat1 = prv1[1]
    phi2 = prv2[0]
    ehat2 = prv2[1]

    phi = 2*np.arccos(np.cos(0.5*phi1)*np.cos(0.5*phi2)-np.sin(0.5*phi1)*np.sin(0.5*phi2)*np.dot(ehat1,ehat2))
    ehat = (1/np.sin(0.5*phi))*(np.cos(0.5*phi2)*np.sin(0.5*phi1)*ehat1+np.cos(0.5*phi1)*np.sin(np.sin(0.5*phi2))+np.sin(0.5*phi1)*np.sin(0.5*phi2)*np.cross(ehat1,ehat2))

    return phi, ehat

def prv_sub(prv1,prv2):
    """ Subtracts two prvs
    prv1,prv2 of format [phi, [ehat]]"""
    phi1 = prv1[0]
    ehat1 = prv1[1]
    phi2 = prv2[0]
    ehat2 = prv2[1]

    phi = 2*np.arccos(np.cos(0.5*phi1)*np.cos(0.5*phi2)+np.sin(0.5*phi1)*np.sin(0.5*phi2)*np.dot(ehat1,ehat2))
    ehat = (1/np.sin(0.5*phi))*(np.cos(0.5*phi2)*np.sin(0.5*phi1)*ehat1-np.cos(0.5*phi1)*np.sin(np.sin(0.5*phi2))+np.sin(0.5*phi1)*np.sin(0.5*phi2)*np.cross(ehat1,ehat2))

    return phi, ehat

def prv2quat(phi,ehat):
    """ Converts prv to quaternion"""
    beta0 = np.cos(0.5*phi)
    beta1 = np.sin(0.5*phi)*ehat[0]
    beta2 = np.sin(0.5*phi)*ehat[1]
    beta3 = np.sin(0.5*phi)*ehat[2]

    quat = np.array([beta0,beta1,beta2,beta3])
    quat = quat/np.linalg.norm(quat) # ensure unit length quaternion

    return quat

def dcm2quat(dcm):
    """Convert dcm to quaternion using Sheppard's method"""
    b0_test = 0.25 * (1 + np.trace(dcm))
    b1_test = 0.25 * (1 + 2 * dcm[0,0] - np.trace(dcm))
    b2_test = 0.25 * (1 + 2 * dcm[1,1] - np.trace(dcm))
    b3_test = 0.25 * (1 + 2 * dcm[2,2] - np.trace(dcm))

    beta_test = np.array([b0_test,b1_test,b2_test,b3_test])
    max_ind = np.argmax(beta_test)

    if max_ind == 0:
        b0 = np.sqrt(b0_test)
        b1 = 0.25 * (dcm[1,2] - dcm[2,1])/ b0
        b2 = 0.25 * (dcm[2,0] - dcm[0,2])/ b0
        b3 = 0.25 * (dcm[0,1] - dcm[1,0])/ b0
    elif max_ind ==1:
        b1 = np.sqrt(b1_test)
        b0 = 0.25 * (dcm[1, 2] - dcm[2, 1]) / b1
        b2 = 0.25 * (dcm[0, 1] - dcm[1, 0]) / b1
        b3 = 0.25 * (dcm[2, 0] - dcm[0, 2]) / b1
    elif max_ind == 2:
        b2 = np.sqrt(b2_test)
        b0 = 0.25 * (dcm[2, 0] - dcm[1, 0]) / b2
        b1 = 0.25 * (dcm[0, 1] + dcm[1, 0]) / b2
        b3 = 0.25 * (dcm[1, 2] + dcm[2, 0]) / b2
    elif max_ind == 3:
        b3 = np.sqrt(b3_test)
        b0 = 0.25 * (dcm[0, 1] - dcm[1, 0]) / b3
        b1 = 0.25 * (dcm[2, 0] + dcm[0, 2]) / b3
        b2 = 0.25 * (dcm[1, 2] + dcm[2, 0]) / b3

    quat = np.array([b0,b1,b2,b3])
    quat = quat/np.linalg.norm(quat) # ensure quaternion is unit length

    return quat

def quat2dcm(quat):
    """Converts a quaternion (numpy array) to a dcm"""
    quat = quat/np.linalg.norm(quat) # make sure quaternion is unit length
    b0,b1,b2,b3 = quat[0], quat[1], quat[2], quat[3]
    dcm = np.array([[b0**2+b1**2-b2**2-b3**2, 2*(b1*b2+b0*b3), 2*(b1*b3-b0*b2)],[ 2*(b1*b2-b0*b3), b0**2-b1**2+b2**2-b3**2, 2*(b2*b3+b0*b1)],
    [2*(b1*b3+b0*b2), 2*(b2*b3-b0*b1), b0**2-b1**2-b2**2+b3**2]])

    return dcm

def quat_add(quat1,quat2):
    """ Adds two quaternions (numpy arrays)"""
    b1 = quat1/np.linalg.norm(quat1)
    b2 = quat2/np.linalg.norm(quat2)

    beta2_mat = np.array([[b2[0],-b2[1],-b2[2],-b2[3]],[b2[1],b2[0],b2[3],-b2[2]],[b2[2],-b2[3],b2[0],b2[1]],[b2[3],b2[2],-b2[1],b2[0]]])

    beta_added = np.matmul(beta2_mat,np.transpose(b1))
    beta_added = beta_added/np.linalg.norm(beta_added)

    return beta_added

def quat_sub(quat1,quat2, var):
    """ Subtracts 2 quaternions , quat1-quat2
    -  quat 1 is the left hand side (the product of 2 quaternions), while q2 is on the right hand side, one of the quaternions being solved gor
    - var represents which quaternion is being solved for (i.e) is we have q = q1*q2
    are we solving for q1 (the right dcm) or q2 (the left dcm)"""
    b1 = quat1 / np.linalg.norm(quat1)
    b2 = quat2 / np.linalg.norm(quat2)

    if var == "first":
        # if the quaternion being solved for is the first rotation
        beta2_mat = np.array(
            [[b2[0], -b2[1], -b2[2], -b2[3]], [b2[1], b2[0], b2[3], -b2[2]], [b2[2], -b2[3], b2[0], b2[1]],
             [b2[3], b2[2], -b2[1], b2[0]]])

    elif var == "second":
        # if the quaternion being solved for is the first rotation
        beta2_mat = np.array(
            [[b2[0], -b2[1], -b2[2], -b2[3]], [b2[1], b2[0], -b2[3], b2[2]], [b2[2], b2[3], b2[0], -b2[1]],
             [b2[3], -b2[2], b2[1], b2[0]]])

    b2_mat_inv = np.linalg.inv(beta2_mat)

    b_final = np.matmul(b2_mat_inv,np.transpose(b1))

    return b_final


def crp2dcm(q):
    """ Converts classical rodrigues parameter to dcm"""
    q_tilde = tilde_matrix(q) # get tilde matrix
    dcm = (1/(1+np.dot(q,q)))*((1-np.dot(q,q))*np.identity(3)+2*np.outer(q,q)-2*q_tilde)
    return dcm

def dcm2crp(dcm):
    """Converts a dcm to a crp"""
    zeta = np.sqrt(np.trace(dcm)+1)
    q = (1/zeta**2)*np.array([dcm[1,2]-dcm[2,1],dcm[2,0]-dcm[0,2],dcm[0,1]-dcm[1,0]])
    return q

def crp2quat(q):
    """ Converts crp to a quaternion"""
    beta0 = 1/(np.sqrt(1+np.dot(q,q)))
    beta1 = q[0]/(np.sqrt(1+np.dot(q,q)))
    beta2 = q[1]/(np.sqrt(1+np.dot(q,q)))
    beta3 = q[2]/(np.sqrt(1+np.dot(q,q)))
    beta = np.array([beta0,beta1,beta2,beta3])
    beta = beta/np.linalg.norm(beta) # normalize quaternion

    return beta


def crp_add(q1,q2):
    """ Adds 2 crps, q1 + q2"""
    q = (1/(1-np.dot(q2,q1)))*(q2+q1-np.cross(q2,q1))
    return q
def crp_sub(q1,q2):
    """ Subtracts 2 crps, q1-q2"""
    q = (1/(1+np.dot(q2,q1)))*(q1-q2+np.cross(q1,q2))
    return q

def mrp2quat(sig):
    """ Converts a mrp to a quaternion"""
    beta0 = (1-np.dot(sig,sig))/(1+np.dot(sig,sig))
    beta1 = 2*sig[0]/(1+np.dot(sig,sig))
    beta2 = 2*sig[1]/(1+np.dot(sig,sig))
    beta3 = 2*sig[2]/(1+np.dot(sig,sig))

    beta = np.array([beta0,beta1,beta2,beta3])
    beta = beta/np.linalg.norm(beta) # check quat is normalized

    return beta
def quat2mrp(beta):
    """ Converts a quaternion to a mrp"""
    sig = (1/(1+beta[0]))*beta[1:]

def mrp2crp(sig):
    """Converts a mrp to crp"""
    q = (2/(1-np.dot(sig,sig)))*sig
    return q

def crp2mrp(q):
    """ Converts a crp to mrp"""
    sig = (1/(1+np.sqrt(1+np.dot(q,q))))*q
    return sig
def prv2mrp(phi,ehat):
    """ Converts prv to mrp"""
    sig = np.tan(0.25*phi)*ehat
    return sig

def dcm2mrp(dcm):
    """ Converts dcm to mrp"""
    zeta = np.sqrt(np.trace(dcm)+1)
    sig = (1/(zeta*(zeta+2)))*np.array([dcm[1,2]-dcm[2,1],dcm[2,0]-dcm[0,2],dcm[0,1]-dcm[1,0]])
    return sig

def mrp2dcm(sig):
    """ converts mrp to a dcm"""
    sig_tilde = tilde_matrix(sig)
    dcm = np.identity(3) + (1/((1+np.dot(sig,sig))**2))*(8*np.matmul(sig_tilde,sig_tilde)-4*(1-(np.dot(sig,sig)))*sig_tilde)
    return dcm

def mrp_add(s1,s2):
    """ Adds 2 mrps s1+s2"""
    sig = ((1-np.dot(s1,s1))*s2+(1-np.dot(s2,s2))*s1-2*np.cross(s2,s1))/(1+np.dot(s1,s1)*np.dot(s2,s2)-2*np.dot(s1,s2))
    return sig

def mrp_sub(s1,s2):
    """ subtracts 2 mrps s2-s1"""
    sig = ((1-np.dot(s1,s1))*s2-(1-np.dot(s2,s2))*s1+2*np.cross(s2,s1))/(1+np.dot(s1,s1)*np.dot(s2,s2)+2*np.dot(s1,s2))
    return sig
















