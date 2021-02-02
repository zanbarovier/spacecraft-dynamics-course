import numpy as np

def int01(inp_func,x0,tspan,dt):
    T = np.arange(tspan[0],tspan[1],dt)
    L = len(T)
    X = np.zeros([L,len(x0)])
    X[0,:] = x0
    for i in range(L-1):
        x_prime = inp_func(X[i,:],T[i])
        X[i+1,:] = X[i,:]+dt*x_prime

    return T,X

def rk4_quat(inp_func,y0,tspan,dt):
    N =  len(y0)
    tout = np.arange(tspan[0],tspan[1]+dt,dt)
    L = len(tout)
    yout = np.zeros([L,N])
    yout[0,:] = y0
    y = y0

    for i in range(L-1):
        time = tout[i]
        # integrations
        ydot1 = inp_func(time,y)
        ydot2 = inp_func(time+0.5*dt,y+ydot1*0.5*dt)
        ydot3 = inp_func(time+0.5*dt, y + ydot2*0.5*dt)
        ydot4 = inp_func(time+dt, y +ydot3*dt)
        ydot_RK4_sum = (1/6)*(ydot1+2*ydot2+2*ydot3+ydot4)
        y_next = y +dt*ydot_RK4_sum
        y_next = y_next/np.linalg.norm(y_next)
        yout[i+1,:] = y_next
        y = y_next

    return tout,yout

def rk4_crp(inp_func,y0,tspan,dt):
    N =  len(y0)
    tout = np.arange(tspan[0],tspan[1]+dt,dt)
    L = len(tout)
    yout = np.zeros([L,N])
    yout[0,:] = y0
    y = y0

    for i in range(L-1):
        time = tout[i]
        # integrations
        ydot1 = inp_func(time,y)
        ydot2 = inp_func(time+0.5*dt,y+ydot1*0.5*dt)
        ydot3 = inp_func(time+0.5*dt, y + ydot2*0.5*dt)
        ydot4 = inp_func(time+dt, y +ydot3*dt)
        ydot_RK4_sum = (1/6)*(ydot1+2*ydot2+2*ydot3+ydot4)
        y_next = y +dt*ydot_RK4_sum
        yout[i+1,:] = y_next
        y = y_next

    return tout,yout

def rk4_mrp(inp_func,y0,tspan,dt):
    N =  len(y0)
    tout = np.arange(tspan[0],tspan[1]+dt,dt)
    L = len(tout)
    yout = np.zeros([L,N])
    yout[0,:] = y0
    y = y0

    for i in range(L-1):
        time = tout[i]
        # integrations
        ydot1 = inp_func(time,y)
        ydot2 = inp_func(time+0.5*dt,y+ydot1*0.5*dt)
        ydot3 = inp_func(time+0.5*dt, y + ydot2*0.5*dt)
        ydot4 = inp_func(time+dt, y +ydot3*dt)
        ydot_RK4_sum = (1/6)*(ydot1+2*ydot2+2*ydot3+ydot4)
        y_next = y +dt*ydot_RK4_sum
        norm_sig =np.linalg.norm(y_next)
        if norm_sig>1:
            y_next = -y_next/(norm_sig**2)
        yout[i+1,:] = y_next
        y = y_next

    return tout,yout



