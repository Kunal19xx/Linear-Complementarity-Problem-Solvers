import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(10)

def PHI(z,n):
    x = np.reshape(z[0:n,0], (n,1))
    y = np.reshape(z[n:2*n,0], (n,1))
    mu = z[-1,0]
    val = x+y-np.sqrt(np.square(x-y)+4*mu*mu)
    return val

def F(z, M, q, n):
    x = np.reshape(z[0:n,0], (n,1))
    y = np.reshape(z[n:2*n,0], (n,1))
    mu = z[-1:,0]
    top = np.matmul(M,x)-y+q
    mid = PHI(z, n)
    bot = mu
    return np.vstack((top,mid,bot))

def jacobian(z, M, q, n):
    x = np.reshape(z[0:n,0], (n,1))
    y = np.reshape(z[n:2*n,0], (n,1))
    mu = z[-1,0]
    top = np.hstack((M,-np.identity(n),np.reshape(np.zeros(n), (n,1))))
    Dx = np.diag(np.squeeze(1-(x-y)/np.sqrt(np.square(x-y)+4*mu**2)))
    Dy = np.diag(np.squeeze(1+(x-y)/np.sqrt(np.square(x-y)+4*mu**2)))
    Dmu = -np.reshape(4*mu/np.sqrt(np.square(x-y)+4*mu**2), (n,1))
    mid = np.hstack((Dx,Dy,Dmu))
    bot = np.reshape(np.zeros(2*n+1), (1,2*n+1))
    bot[0][-1] = 1
    val = np.vstack((top,mid))
    return np.vstack((val,bot))

def corrector(u_hat, del_u_hat, alpha2, i, beta):
    n = u_hat.shape[0]//2
    lambd = alpha2**i
    intmd = del_u_hat
    intmd[-1,0] = -sigma*u_hat[-1,0]
    return np.linalg.norm(PHI(u_hat+intmd*lambd,n)) > (1- sigma*lambd)*u_hat[-1,0]*beta

def pred(u, del_u, alpha1, t, beta):
    b = u +del_u
    b[-1,0] *= alpha1**t
    return np.linalg.norm(b) <= b[-1,0]*beta

def plotter(arr):
    l2 = np.array(arr)
    plt.xlabel('iteration count')
    plt.ylabel('Norm PHI')
    plt.plot(l2[:,0],l2[:,1])
    plt.show()
    return None

def show_results(arr,n):
    x = np.reshape(arr[0:n,0], (n,1))
    y = np.reshape(arr[n:2*n,0], (n,1))
    z1 = np.hstack((x,y)) 
    print(pd.DataFrame(np.round(z1,4), columns = ['x','y']))
    return None

def main(M, q, max_itr, alpha1, alpha2, sigma):
    n = M.shape[0]
    mu = 1
    
    #One initialization.
    #x = np.reshape(np.ones(n),(n,1))*2
    
    #Normal Initialization
    x = np.reshape(np.random.normal(0,0.5*sigma, n), (n,1))
    
    #Uniform initialization
    #x = np.reshape(np.random.uniform(-1,1, n), (n,1))

    y = np.matmul(M,x)+q
    z = np.vstack((x,y,mu))
    
    # mu estimation
    while np.max(PHI(z,n)) >= 0:
        z[-1,0] += 0.001
        mu += 0.001

    # beta estimation
    beta_low = 2*np.sqrt(n)
    beta_up = np.linalg.norm(PHI(z, n))/mu
    beta = max(beta_low ,beta_up) + 20

    print(beta_up, beta_low, beta, mu)

    itr_ct = 1
    li = []

    while itr_ct <= max_itr:

        # predictor step
        jack = jacobian(z, M, q, n)
        del_z = -np.matmul(np.linalg.inv(jack),F(z, M, q, n))
        del_z[-1,0] = 0
        if np.linalg.norm(PHI(z + del_z,n)) > beta*mu:
            z_hat = z
        else :
            t = 0
            while pred(z, del_z, alpha1, t, beta):
                t += 1
            if t != 0:
                t -= 1
            z_hat = z + del_z
            z_hat[-1,0] *= alpha1**t

        #corrector step
        C_top =np.reshape(np.zeros(2*n),(2*n,1))
        C_bot =(1-sigma)*z_hat[-1,0]
        C =np.vstack((C_top,C_bot))
        del_z_hat =np.matmul((np.linalg.inv(jacobian(z_hat,M,q,n))),(C-F(z_hat,M,q,n)))
        i = 0
        while corrector(z_hat, del_z_hat, alpha2, i, beta) :
            i += 1
        lambd = alpha2**i
        del_z_hat[-1,0] = 0
        z = z_hat + lambd*del_z_hat
        z[-1,0] = (1-sigma*lambd)*z_hat[-1,0]
        itr_ct += 1

        # caching for plot
        li.append((itr_ct,np.linalg.norm(PHI(z,n))))
    show_results(z,n)
    return z, li


#epsilon = 0.00001
df = pd.read_csv('ORq1.csv', header=None)
M = np.array(df.iloc[:,:-1])
q = np.array(df.iloc[:,-1:])
n = M.shape[0]
max_itr = int(input('Max_iteration is \n'))
sigma = 0.8
alpha1 = 0.1
alpha2 = 0.01

z,li = main(M, q, max_itr, alpha1, alpha2, sigma)

plotter(li)
print('\nLast ||PHI|| value is  = ', np.linalg.norm(PHI(z,n)))