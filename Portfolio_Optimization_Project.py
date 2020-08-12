import numpy as np
from   scipy.sparse import spdiags, identity
from   scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
#from   bokeh.plotting import figure, output_notebook, show, gridplot, save
import pandas as pd
import math
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator
%matplotlib inline  

# Useful Utility functions
# simple testing
def testVariance(gamma,Wmin,Wmax,Pmin,Pmax,ifAllowBrt = True):
    W, V, U, ctrls = main(gamma, Wmin,Wmax,Pmin,Pmax,ifAllowBrt, -1)
    Vvalue = interpolation(W,V,1)
    Uvalue = interpolation(W,U,1)
    var= Vvalue + gamma * Uvalue - gamma**2 /4 - Uvalue**2
    print('var is {}  and U is {}'.format(var,Uvalue))
    
# Monte Carlo simulation with controls
def simulatePaths(r, sigmaxi, W0, pi, sigma,ctrls, numOfPath):
    increZ = np.random.normal(0, 1, numOfPath*len(ctrls)).reshape(numOfPath,len(ctrls))
    Wealth = np.zeros_like(increZ)
    for j in range(numOfPath):
        for i in range(len(ctrls)):
            if i == 0:
                increW = ((r + ctrls[i]*sigmaxi)*W0 + pi)*dt + ctrls[i]*sigma*W0*increZ[j][i]
                Wealth[j][i] = W0+ increW
            else:
                increW = ((r + ctrls[i]*sigmaxi)*Wealth[j][i-1] + pi)*dt + ctrls[i]*sigma*Wealth[j][i-1]*increZ[j][i] 
                Wealth[j][i] = Wealth[j][i-1]+ increW
    return Wealth[:,-1]

# simulate GBM process
# :return log returns
def simulateReturnPathsGBM(r, sigma, xi, dt, numOfPath, timesteps):
    S0 = 1
    sigmaxi = sigma*xi
    timeSteps = len(timesteps)
    increZ = np.random.normal(0, 1, numOfPath * timeSteps).reshape(numOfPath,timeSteps)
    returns = np.zeros_like(increZ)
    for j in range(numOfPath):
        for i in range(timeSteps):
                if i == 0:
                    returns[j][i] = 0
                else:
                    returns[j][i] = (r + sigmaxi - sigma**2/2) * timesteps[i] + sigma * increZ[j][i] 

    return returns

# simulate paths backward
def simulatePathsBackwardStrategy(wealth, intiCtrls:list, returns:list, C, r, dt, numOfPath,timesteps):
    W0 = 1
    for j in range(numOfPath):
        for i,t in enumerate(timesteps):
            if i == 0:
                wealth[j][i] = W0 * ( intiCtrls[j][i] * returns[j][i] + r ) + C * dt
            else:
                wealth[j][i] = wealth[j][i-1] * ( intiCtrls[j][i] * returns[j][i] + r ) + C * dt
    return wealth

# analytical solution for allow bankruptcy case
# params initialize later
# for testing
def analyticSolution(gamma):
    lastPart = np.exp(-r*(T-t) + xi**2 * T)/(2* gamma)
    optP = -xi/(sigma * W) * (W - (W0*np.exp(r*t) + pi/r * (np.exp(r*t) - 1)) - lastPart)
    termWealth = simulatePaths(r, sigmaxi, W0, pi, sigma,optP,100)
    exptW = W0 * np.exp(r*T) + pi* (np.exp(r*T) - 1)/r + np.sqrt(np.exp(xi**2 * T) - 1)  * np.std(termWealth)
    lamb_da = 0.5* math.pow(gamma/2 - exptW, -1)
    varW = np.exp(xi**2 * T - 1)/(4* lamb_da**2)

# partition paths randomly
def randomPartition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

# partition based on the value similarity
def wealthPartition(pathsValues :list, n):
    numInBundle = int(len(pathsValues)/n)
    order = sorted(range(len(pathsValues)), key=lambda k: pathsValues[k])
    return [order[x:x + numInBundle] for x in range(0, len(order), numInBundle)]

# regression using basis function
# https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
def quadraticFit(x,y,degree):
    polynomial_features= PolynomialFeatures(degree)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    return model

# costomized interpolation method
def interpolation(myArrayOne:np.array, myArrayTwo:np.array, value):
    leftIdx = myArrayOne.tolist().index(min(myArrayOne, key=lambda x:abs(x-value)))
    rightIdx = leftIdx + 1
    proportion = (value - myArrayOne[leftIdx])/(myArrayOne[rightIdx] - myArrayOne[leftIdx])
    return myArrayTwo[leftIdx] + proportion* (myArrayTwo[rightIdx] - myArrayTwo[leftIdx])



def backwardStrategyTest(gamma):
    r = 0.03
    sigma = 0.15
    xi = 0.33
    C = 0.0
    T = 2
    W0 = 1
    M = 80
    numOfPath = 5000
    numOfBundles = 20
    dt = T/M
    # Xmax = 1 + C*dt / 
    # every path the same? no
    timesteps = np.linspace( 0.0, T, M + 1 )[:-1]
    intiCtrls = np.random.uniform(0,1.5,M * numOfPath ).reshape( numOfPath, M)
    returns = simulateReturnPathsGBM(r, sigma, xi,dt, numOfPath,timesteps)
    returns = np.exp(returns) - r
    wealth = np.zeros_like(returns)
    wealthProcess = simulatePathsBackwardStrategy(wealth,intiCtrls,returns, C, r, dt, numOfPath,timesteps)
    initContiValue = list(map(lambda x : x**2, wealthProcess[:,-1] - gamma/2 ))
    
    # bundlePaths = dict([(str(idx) + '_' + str(i) ,wealthProcess[i,:])for idx,bundle in enumerate(bundles) for i in bundle]) 
    optPs = [0 for i in range(numOfBundles)]
    

    # backward recursion
    for i in range(len(timesteps)-2,-1,-1):
 
        contiValuetList = np.zeros_like(initContiValue)
        bundles = wealthPartition(wealthProcess[:,i].tolist(), numOfBundles)
        for idx , bundle in enumerate(bundles):
            optPs[idx] = []
            wealthValueInBundle = []
            contiValueInBundle = []
            for path in bundle:
                if i == len(timesteps)-1:
                    wealthValueInBundle.append(wealthProcess[path][i+1])
                    contiValueInBundle.append(initContiValue[path])
                else:
                    wealthValueInBundle.append(wealthProcess[path][i+1])
                    contiValueInBundle.append(contiValuetList[path])

            # this is the same for every paths in a bundle
            # t+dt
            localApprox = quadraticFit(np.array(wealthValueInBundle),np.array(contiValueInBundle),2)

            for path in bundle:
                # t+dt
                # fist order condition 
                xhat = localApprox.coef_[0][1] + 2* wealthProcess[path][i+1]
                optPs[idx].append(xhat) 
                # step 2.3
                temp = xhat * returns[path][i+1] + r 
                Jhat = (wealthProcess[path][i+1] * temp - gamma/2)**2
                # step 3
                # t
                tempt = intiCtrls[path][i] * returns[path][i] + r 
                Jtilda = (wealthProcess[path][i]*tempt - gamma/2)**2
                #contiValuetList.append(contiValuet)

                if Jtilda > Jhat:
                    #print("true")
                    updatedCtrl = xhat
                else:
                    #print('False')
                    updatedCtrl = intiCtrls[path][i]

                tempt = updatedCtrl * returns[path][i] + r 
                J = (wealthProcess[path][i] * tempt - gamma/2)**2
                contiValuetList[path] = J
    return contiValuetList, wealthProcess

gammaList = np.linspace(9.125, 20, 80)
#gammaList = [0.1]
VbackList = []
UbackList = []
x = []
for gamma in gammaList:
    contiValuetList, wealthProcess= backwardStrategyTest(gamma)
    #V= np.mean(contiValuetList)
    V= contiValuetList[-1]
    U = wealthProcess[:,0][-1]
    #U = np.mean(wealthProcess[:,0])
    var = V + gamma * U- gamma**2 /4 - U**2
    if  var > 0:
        VbackList.append(V)
        UbackList.append(U)  
        std = np.sqrt(V + gamma * U- gamma**2 /4 - U**2)
        #std = V + gamma * U- gamma**2 /4 - U**2      
        x.append(std) 
plt.plot(x,UbackList)

xold = x.copy()
yold = UbackList.copy()


# Cong and Oos (2016)
# multi-period
def multiStageStrategy(gamma):
    S0 = 1
    r  = 0.03 #since r is log return
    xi = 0.33
    sigma = 0.15
    C = 0
    numOfPath = 5000
    #gamma = 14.47
    #gamma = 40
    M = 80 # rebalancing opportunities
    T = 20 
    dt = T / M
    W0 = 1
    Pmax = 1.5
    Pmin = 0

    timesteps = np.linspace( 0.0, T, M + 1 )[:-1]
    returns = simulateReturnPathsGBM(r, sigma, xi,dt, numOfPath,timesteps)
    returns = np.exp(returns) - r
    wealth = np.zeros_like(returns)
    optPs = np.zeros_like(returns)

    #returns = np.exp(returns) - r

    def simulatePathForward(gamma,numOfPath):
        for j in range(numOfPath):
            for i,t in enumerate(timesteps):
                if i == 0:
                    denom = W0 *r **((T-t)/dt - 1) *np.mean(returns[:,i]**2)
                    #print(denom)
                    optP = (gamma/2 - W0 *r **((T-t)/dt))* np.mean(returns[:,i])/ denom 
                    if optP < 1.5 and optP > 0:
                        optP = optP 
                    elif optP > 1.5:
                        optP = 1.5
                    else:
                        optP = 0

                    optPs[j][i] = optP
                    wealth[j][i] = W0 * (optP*returns[j][i] + r ) + C * dt
                else:
                    denom = wealth[j][i-1] *r **((T-t)/dt - 1) *np.mean(returns[:,i]**2)

                    optP = (gamma/2 - wealth[j][i-1] *r **((T-t)/dt))* np.mean(returns[:,i])/ denom 
                    
                    if optP < Pmax and optP > Pmin:
                        optP = optP 
                    elif optP > Pmax:
                        optP = Pmax
                    else:
                        optP = 0

                    optPs[j][i] = optP
                    wealth[j][i] = wealth[j][i-1] * (optP*returns[j][i] +r) + C * dt
        return wealth[:,-1]
    
    termWealth = simulatePathForward(gamma,numOfPath)
    return termWealth,optPs

stdFwdList =[]
muFwdList = []
gammaList = np.linspace(9.125,20, 80)

for gamma in gammaList:
    termWealth,optPs = multiStageStrategy(gamma)
    stdFwdList.append(np.std(termWealth))
    muFwdList.append(np.mean(termWealth))

# Plots Cong (2016)
# multi-stage alone
fig4,ax = plt.subplots()
plt.plot(stdFwdList,muFwdList,label = 'Multi-Stage Strategy')
ax.legend(loc='upper left')
ax.set_title('Efficient Frontier for Multi-Stage Strategy')
ax.set_xlabel('std[W(t=T)] at t = 0')
ax.set_ylabel('E[W(t=T)] at t = 0')
plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.show()

# backward iteration alone
fig5,ax = plt.subplots()
plt.plot(x, UbackList,color = 'green', lw=2, label = '1 Backward Iteration')
ax.legend(loc='upper left')
ax.set_title('Efficient Frontier for Backward Recursion Strategy')
ax.set_xlabel('std[W(t=T)] at t = 0')
ax.set_ylabel('E[W(t=T)] at t = 0')
plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.show()

# multi-stage vs 1 backward iteration
fig6,ax = plt.subplots()
plt.plot(stdFwdList,muFwdList,label = 'Multi-stage')
plt.plot(x, UbackList, color = 'green', lw=2,label = '1 Backward Iteration')
ax.legend(loc='upper left')
ax.set_title('Efficient Frontier Comparison')
ax.set_xlabel('std[W(t=T)] at t = 0')
ax.set_ylabel('E[W(t=T)] at t = 0')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Wang and Peter (2010) main function
def main(gamma, Wmin,Wmax, Pmin, Pmax, allowBankrupt,iteration):
    # Params initialization
    r     = 0.03
    sigma = 0.15
    xi    = 0.33
    pi    = 0.1
    W0    = 1.0
    T     = 20.0
    #gamma = 14.47
    #Wmin  = 0.0
    #Wmax  = 5.0
    #M     = 1600
    #N     = 100
    M     = 728
    N     = 160
    tol   = 1e-6
    scale = 1.0
    #Pmin = 0.0
    #Pmax  = 1.5
    J     = 10
    hsigsq  = 0.5 * sigma ** 2 
    sigmaxi = sigma * xi
    dW      = ( Wmax - Wmin) / N
    dt      = T / M
    dWsq    = dW ** 2  
    W    = np.linspace( Wmin, Wmax, N + 1 ) #  N steps
    Ps   = np.linspace( Pmin, Pmax, J ) # discretize controls
    I    = identity( N + 1 )
    Gn   = np.zeros_like( W )
    Gnp1 = np.zeros_like( W )
    Hn   = np.zeros_like( W )
    Hnp1 = np.zeros_like( W )
    terminal_values_V = ( W - 0.5 * gamma ) ** 2
    terminal_values_U = W
    def bc( t ,allowBankrupt = True): # boundary condition
        tau = T - t
        c   = ( 2 * pi ) / r
        e1 = np.exp( r * tau )
        e2 = np.exp( 2 * r * tau )
        alpha = e2 * ( Wmax**2 )
        beta  = ( c * e2 - ( gamma + c ) *e1 ) * Wmax
        delta = ( ( gamma**2 ) / 4.0 ) +  ( ( pi * c ) / ( 2 * r ) ) * ( e2 - 1 )\
                - ( ( pi * ( gamma + c ) ) / r ) * ( e1 - 1 )  
        return alpha + beta + delta

    def bcV(t, allowBankrupt = True):
        if allowBankrupt:
            tau = T - t
            p = - xi /sigma
            k1 = r + p*sigmaxi
            k2 = (p*sigma)**2
            H1 = np.exp((2*k1 + k2)* tau) * Wmax**2
            return H1
        else:
            tau = T - t
            p = - xi /sigma
            k1 = r 
            k2 = 0
            H1 = np.exp((2*k1 + k2)* tau) * Wmax**2
            return H1

    def bcU( t, allowBankrupt = True ):
        if allowBankrupt:
            tau = T - t
            p = - xi /sigma
            J1 = np.exp((r+ p*sigmaxi)* tau )* Wmax      
            J2 =  0
            return J1 + J2

        else:

            tau = T - t
            p = - xi /sigma
            J1 = np.exp( r* tau )* Wmax      
            J2 =  0
            return J1 + J2

    def alpha( W, p, dirn = CENTRAL ):
        t1 = hsigsq * (p**2) * (W**2) / dWsq
        t2 = ( pi + W * ( r + p * sigmaxi ) ) 
        if dirn == CENTRAL:
            return t1 - t2 / ( 2 * dW )
        elif dirn == BACKWARD:
            return t1 - t2 / dW
        elif dirn == FORWARD:
            return t1
    
    def beta( W, p, dirn = CENTRAL ):
        t1 = hsigsq * (p**2) * (W**2) / dWsq
        t2 = ( pi + W * ( r + p * sigmaxi ) ) 
        if dirn == CENTRAL:
            return t1 + t2 / (2 *dW)
        elif dirn == FORWARD:
            return t1 + t2 / dW
        elif dirn == BACKWARD:
            return t1
    
    def makeDiagMat( alphas, betas ):
        d0, dl, d2 = -( alphas + betas ), np.roll( alphas, -1 ), np.roll( betas, 1 )
        d0[-1] = 0.
        dl [-2:] = 0.
        data = np.array( [ d0, dl, d2 ] )
        diags = np.array( [ 0, -1, 1 ] )
        return spdiags( data, diags, N + 1, N + 1 )
    
    
    def find_optima1_ctrls( Vhat, t , allowBankrupt):
        if allowBankrupt:
            W0 = 1           
            lastPart = np.exp(-r*(T-t) + xi**2 * T)/(2* gamma)
            optP = -xi/(sigma * W) * (W - (W0*np.exp(r*t) + pi/r * (np.exp(r*t) - 1)) - lastPart)
            
            optdiffs = [1 for i in range(len(optP))]
            return optP, optdiffs

        else:

            Fmin = np.tile( np.inf, Vhat.size )

            optdiffs = np.zeros_like( Vhat, dtype = np.int )
            optP    = np.zeros_like( Vhat )          
            alphas  = np.zeros_like( Vhat ) # the final
            betas   = np.zeros_like( Vhat ) # the final
            curDiffs = np.zeros_like( Vhat, dtype = np.int )
            
            for p in Ps: # Hnd the optimal control
                alphas[:] = -np.inf
                betas[:] = -np.inf
                curDiffs[:] = CENTRAL
                
                for diff in [ CENTRAL, FORWARD, BACKWARD ]:
                    a = alpha( W, p, diff)
                    b = beta( W, p, diff )
                    positive_coeff_indices = np.logical_and( a >= 0.0, b >= 0.0 ) == True
                    positive_coeff_indices = np.logical_and( positive_coeff_indices, alphas== -np.inf )
                    indices = np.where( positive_coeff_indices )

                    alphas[ indices ] = a[ indices ]
                    betas[ indices ] = b[ indices ]
                    curDiffs[ indices ] = diff
                    
                M = makeDiagMat( alphas, betas )
                F = M.dot( Vhat )
                indices = np.where( F < Fmin )
                
                Fmin[indices] = F[indices ]
                optP[indices] = p
                optdiffs[indices] = curDiffs[ indices ]
            return optP, optdiffs
    
    timesteps = np.linspace( 0.0, T, M + 1 )[:-1] # drop last item which is T=20.0
    timesteps = np.flipud( timesteps )
    V = terminal_values_V
    U = terminal_values_U
    alphas = np.zeros_like( V )
    betas = np.zeros_like( V )

    for t in timesteps:
        # policy iteration algorithm
        ite = 0
        Vhat = V.copy()
        Uhat = U.copy()
        Gnp1[-1] =bc(t+ dt, allowBankrupt)
        Gn[-1] = bc( t, allowBankrupt)
        Hnp1[-1] = bcU(t+dt, allowBankrupt)
        Hn[-1] = bcU(t, allowBankrupt)     
        B   = Gn - Gnp1 # new boundary cone - old boundary cone
        BU  = Hn - Hnp1
        while True:
            ctrls, diffs = find_optima1_ctrls( Vhat, t ,allowBankrupt)
            for diff in [ CENTRAL, FORWARD, BACKWARD ]:
                indices = np.where( diffs == diff )
                alphas[indices] = alpha( W[indices], ctrls[indices], diff )
                betas[indices] = beta( W[indices], ctrls[indices], diff )
                
            A   = makeDiagMat( alphas, betas )
            M = I - dt * A
            # M contains info of P*
            Vnew = spsolve( M, V + B )
            Unew = spsolve( M, U + BU )
            scale    = np.maximum( np.abs( Vnew ), np.ones_like( Vnew ) )
            scale_U    = np.maximum( np.abs( Unew ), np.ones_like( Unew ) )
            residuals = np.abs( Vnew - Vhat ) / scale
            residuals_U = np.abs( Unew - Uhat ) / scale_U
            if np.all( residuals[:-1] < tol ) and ite > iteration:
                V = Vnew
                U = Unew
                break
            else:
                Vhat = Vnew
                Uhat = Unew
                ite += 1
    return W, V, U, ctrls

# plots - Wang and Forysth (2010)
#plot effinicent frontier  - bounded ctrl & bankrupt allowed
W0 = 1
r  = 0.03
xi = 0.33
T  = 20
VarListAllBrt =[]
UListAllBrt =[]
VListAllBrt =[]
ctrlsListAllBrt = []
VarListNoAllBrt =[]
UListNoAllBrt =[]
VListNoAllBrt =[]
ctrlsListNoAllBrt = []
gammaSteps = np.linspace( 0.1, 12, 20)
#gammaSteps = [2,3]
for gamma in gammaSteps:
    # bdd ctrls
    W, V, U, ctrls = main(gamma,0,20, False, 2)
    var = V[0] + gamma * U[0] - gamma**2 /4 - U[0]**2
    VarListAllBrt.append(var) 
    UListAllBrt.append(U[0])
    VListAllBrt.append(V[0])
    ctrlsListAllBrt.append(ctrls)
    # allow brt
    W, V, U, ctrls = main(gamma,0,20, True, 2)
    termWealth = simulatePaths(r, sigmaxi, W0, pi, sigma,ctrls,100)
    exptW = W0 * np.exp(r*T) + pi* (np.exp(r*T) - 1)/r + np.sqrt(np.exp(xi**2 * T) - 1)  * np.std(termWealth)
    lamb_da = 0.5* math.pow(gamma/2 - exptW, -1)
    varW = np.exp(xi**2 * T - 1)/(4* lamb_da**2)
    VarListNoAllBrt.append(varW)
    UListNoAllBrt.append(U[0])
    VListNoAllBrt.append(V[0])
    #ctrlsListNoAllBrt.append(ctls)
    
y =[]
x =[]
mask =[]
for var,mu in zip(VarListNoAllBrt,UListNoAllBrt):
    if var > 0:
        mask.append(True)
        x.append(np.sqrt(var))
        y.append(mu)
    else:
        mask.append(False)
xAllbrt = np.sqrt(VarListAllBrt)
xAllbrt = yAllbrt(mask).tolist()
yAllbrt = UListAllBrt
plt.plot(x,y)
plt.plot(xAllbrt,yAllbrt)

# visualize control process

W, V, U, ctrls = main(9.125,0,5,0,1.5,False,-1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
ax1.plot(W, V, color = 'blue', lw=2)
ax1.set_title('Plot of $E_{p^*}^{t=0}[(W_T - \gamma/2)^2]$ at W = 1, t = 0')
ax2.plot(W, ctrls, color = 'green', lw=2, alpha = 0.5)
fig1 = ax2.set_title('Plot of optimal control $p^*$ against wealth W')

#bdd ctrl

pi = 0.1
x = [np.sqrt(var) for var in VarListAllBrt]
y = UListAllBrt
xAlbrt = np.linspace( 0.8, 1.5, 20)
intersect = W0 * np.exp(r*T) + pi* (np.exp(r*T) - 1)/r + np.sqrt(np.exp(xi**2 * T)- 1)
exptW = list(map(lambda m : intersect + np.sqrt(np.exp(xi**2 * T) - 1) * m, xAlbrt))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
ax1.plot(x, y, color = 'blue', lw=2)
ax1.set_title('Bounded control efficient frontier with $\gamma \in [0.1,50]$ ')
ax2.plot(xAlbrt, exptW, color = 'green', lw=2, alpha = 0.5)
fig1 = ax2.set_title('Allow bankrupcy')

# no bkrt
stdListNoAllBrt =[]
UListNoAllBrt =[]
VListNoAllBrt =[]
ctrlsListNoAllBrt = []
gammaSteps = np.linspace( 0.1, 50, 20)
#gammaSteps = [2,3]
for gamma in gammaSteps:
    # bdd ctrls
    W, V, U, ctrls = main(gamma,0,20,0,5, False, -1)
    Vvalue = interpolation(W,V,1)
    Uvalue = interpolation(W,U,1)
    std = np.sqrt(Vvalue + gamma * Uvalue - gamma**2 /4 - Uvalue**2)
    stdListNoAllBrt.append(std) 
    UListNoAllBrt.append(Uvalue)
    VListNoAllBrt.append(Vvalue)
    ctrlsListAllBrt.append(ctrls)
plt.plot(stdListNoAllBrt,UListNoAllBrt)

# comparison between no bankruptcy and bounded control
fig2,ax = plt.subplots()
plt.plot(stdListNoAllBrt,UListNoAllBrt,label = 'No bankruptcy')
plt.plot(x, y, color = 'green', lw=2,label = 'Bounded control')
ax.legend(loc='upper left')
ax.set_title('Efficient Frontier Comparison')
ax.set_xlabel('std[W(t=T)] at t = 0')
ax.set_ylabel('E[W(t=T)] at t = 0')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()