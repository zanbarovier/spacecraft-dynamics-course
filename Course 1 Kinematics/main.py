import numpy as np
import attitude_coords_library as cl
import ode_integrators as ode
import attitude_diff_eqs as de
import attitude_det_methods as ad
import sympy as sym


v1B = np.array([0.8273 ,0.5541, -0.0920])
v2B = np.array([-0.8285 ,0.5522, -0.0955])
vec_B = np.array([v1B,v2B])
v1N = np.array([-0.1517 ,-0.9669 ,0.2050])
v2N = np.array([-0.8393 ,0.4494 , -0.3044])
vec_N = np.array([v1N,v2N])
w = [5,0.5]

dcm = ad.olae(v1B,v2B,v1N,v2N,w)
print(dcm)





















