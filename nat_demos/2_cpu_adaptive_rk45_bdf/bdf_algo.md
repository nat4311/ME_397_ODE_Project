Note: dxdt is f(x,p), no time component
we have a single g (the ode function, dxdt = f(x,p))
also have single Jg (jacobian of the ode function)
we have a collection of {p_i, x0_i} -> length s. use BDF2_single to solve ODE i
we will use the same t0, t_end, h0 for all
x is vector of size nx1
p is vector of size mx1

need to run many BDF2_single_noJg in parallel on GPU

# BDF2_single_noJg(g, x0, t0, t_end, p, h0=.01)
// g is the ODE function g(x,p) = [xdot1, xdot2, ..., xdotn]
// x0 is the initial nx1 state vector at time t0   [x1, x2, ..., xn]
// t0 and t_end are the time bounds of solution (float)
// Jg is the jacobian function Jg(x) = [nxn matrix]
// h0 is the initial timestep (float)

## Initialization
1.  n = x0.size()
    xcurr = x0
    xprev = None
    xprev2 = None
    xprev3 = None
    t = t0
    h = h0
    xsol = [x0]
    tsol = [t0]
    I = eye(n)

## first step use BDF1
2.  // Initialize for newton's method
    q = xcurr
    Jg = zeros(n,n)
3.  // Newton's method iterations - Note: this could take different no. iters for different threads
    for i in range(newtonMaxIters): 
        residual = q - h*g(q,p) - xcurr
        if norm(residual) < newtonTolerance:
            break
        for j in range(n):
            dq = ones(n)*1e-10
            Jg[:,j] = (g(q+dq, p) - g(q,p)) / 1e-10
        dfinv = (I - h*Jg).inverse()
        q -= dfinv*residual
4.  // update values and save data
    xprev = xcurr
    xcurr = q
    t += h
    xsol.append(xcurr)
    tsol.append(t)
    hprev = h

## all other steps use BDF2
5.  // Initialize for newton's method
    wn = h/hprev
    a = (1+2*wn)/(1+wn)
    b = -(1+wn)^2/(1+wn)*xcurr + (wn^2)/(1+wn)*xprev
    c = a*I
    q = xcurr
    Jg = zeros(n,n)
6.  // Newton's method iterations - Note: this could take different no. iters for different threads
    for i in range(newtonMaxIters): 
        residual = a*q - h*g(q,p) + b
        if norm(residual) < newtonTolerance:
            break
        for j in range(n):
            dq = ones(n) * 1e-10
            Jg[:,j] = (g(q+dq, p) - g(q,p)) / 1e-10
        dfinv = (c - h*Jg).inverse()
        q -= dfinv * residual
7.  // update values and save data
    xprev3 = xprev2
    xprev2 = xprev
    xprev = xcurr
    xcurr = q
    t += h
    xsol.append(xcurr)
    tsol.append(t)
    hprev = h
8.  // update step size h
    if xprev3 not None:
        LTE_norm = norm(1/3*xcurr - xprev + xprev2 - 1/3*xprev3)
        if LTE_norm < 1e-10:
            h = hprev * Fmax
        else:
            sc = Atol + max(np.linalg.norm(xn), np.linalg.norm(xnm1))*Rtol
            err = LTE_norm/sc
            h = hprev * min(Fmax, max(Fmin, (1/err)^(1/3)*F))

## output
9. return xsol, tsol
