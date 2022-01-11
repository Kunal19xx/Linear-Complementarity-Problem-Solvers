import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(10)

def phi(z,A,q,n,s):
    x = np.reshape(z[:,0],(n,1))
    y = np.reshape(z[:,1],(n,1))
    j = y-np.matmul(A,x)-q
    val = (n+s)*np.log(np.matmul(x.T,y)+np.square(np.linalg.norm(j)))
    val -= sum(np.log(x*y))
    return val

def f1(z,A,q,n):
    x = np.reshape(z[:,0],(n,1))
    y = np.reshape(z[:,1],(n,1))
    val = np.matmul(x.T,y)+sum(np.square(y-np.matmul(A,x)-q))
    return val

def f2(z,inc,epsilon):
    if np.linalg.norm(inc) < epsilon == True:
        z += inc
        if 1 in z[:,0]*z[:,1] :
            return True
        else :
            return False
    else:
        return False 

def cost_vs_itr_ct(A, q, max_itr1):
    z,itr_ct, costs = main(A, q, s = 150,epsilon = 0.00001,alpha = 0.05,\
                    beta_bar = 0.99,max_itr=max_itr1, sigma = 0.025,rho = .003)
    li = np.array(costs, dtype= object)
    plt.title('cost vs itr_ct')
    plt.xlabel('itr_ct')
    plt.ylabel('cost')
    plt.scatter(li[:,0],li[:,1])
    plt.show()
    return None

def itr_ct_vs_s(A,q, num):
    li = []
    for i in range(1,num,1):
        _, itr_ct, _ = main(A, q, s = i,epsilon = 0.00001,alpha = 0.05,\
                    beta_bar = 0.99,max_itr=1000, sigma = 0.025,rho = .003)
        li.append([i, itr_ct])
    li = np.array(li, dtype= object)
    plt.title('itr_ct vs s')
    plt.xlabel('s')
    plt.ylabel('itr_ct')
    plt.plot(li[:,0],li[:,1], 'bo')
    plt.show()
    return None
    
def main(A,q,s = 50,epsilon = 0.00001,alpha = 0.7,beta_bar = 1,\
            sigma = 0.001,rho = .02, max_itr = 1000):
    
    n = A.shape[0]
    e = np.reshape(np.ones(n),(n,1))
    itr_ct = 0
    cost_list = []
    
    #initialization Step 0
    x = np.reshape(np.ones(n),(n,1))
    y = np.reshape(np.ones(n),(n,1))
    z = np.hstack((x,y))
    
    while f1(z,A,q,n) > epsilon :    #Step 3 condition 1 check

        cost_list.append([itr_ct, f1(z,A,q,n)])

        x1 = np.squeeze(z[:,0])
        y1 = np.squeeze(z[:,1])
        x_mat_itr = np.diag(x1)
        y_mat_itr = np.diag(y1)
        x = np.reshape(x1,(n,1))
        y = np.reshape(y1,(n,1))

        lhs1 = np.hstack((y_mat_itr,x_mat_itr))
        lhs2 = np.hstack((-A,np.identity(n)))
        lhs = np.vstack((lhs1,lhs2))

        #beta  = np.random.uniform(0,beta_bar)
        beta = 0.003

        rhs1 = -np.matmul(x_mat_itr,y)+beta*np.matmul(x.T,y)*e/n
        rhs2 = -y + np.matmul(A,x)+q
        rhs = np.vstack((rhs1,rhs2))

        delta = np.matmul(np.linalg.inv(lhs),rhs)
        delta_x,delta_y = np.vsplit(delta,2)
        d = np.hstack((delta_x,delta_y))

        # Step 2 begins
        L = 0
        val1 = 100
        val2 = -100
        while val1 > 0 or val2 < 0:
            inc = sigma*pow(rho, L)*d
            val2 = np.min(z+inc)
            if val2 > 0:
                val1 = phi(z+inc,A,q,n,s) - phi(z,A,q,n,s)
                val1 += alpha*sigma*pow(rho, L)*(1-beta)*s
            L += 1
        L -= 1
    
        inc = sigma*pow(rho, L)*d
        itr_ct += 1
        
        #Step 3 condition 2 check
        if f2(z,inc,epsilon) == True or itr_ct >= max_itr:
            z += inc
            break
        z += inc
        
    return z,itr_ct , cost_list

if __name__ == "__main__":
    
    #input is in form of [A|q] matrix
    #st = input('Please enter input file name')
    st = 'ORq1.csv'
    df = pd.read_csv(st, header=None)
    A = np.array(df.iloc[:,:-1])
    q = np.array(df.iloc[:,-1:])
    n = A.shape[0]
    
    z,itr_ct, _ = main(A, q, s = 15000,epsilon = 0.00001,alpha = 0.05,\
                    beta_bar = 0.99,max_itr=10000, sigma = 0.025,rho = .003)
    print (pd.DataFrame(np.round(z,4), columns=(['x','y'])), '\n',  \
           'Iteration count is = ',itr_ct)

    # cost vs itr_ct
    cost_vs_itr_ct(A, q, max_itr1 =50)
    # s vs itr_ct
    itr_ct_vs_s(A,q, num = 25)
