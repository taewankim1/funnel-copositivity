from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
from cvxpy import vec
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

class funlopt_by_COP :
    def __init__(self,ix,iu,iq,ip,iw,N,delT,myScaling,myModel,max_iter=5,
        lambda_mu=0.1,
        w_tr=1,flag_nonlinearity=True) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.delT = delT
        self.N = N
        self.small = 1e-6
        self.w_tr = w_tr
        self.flag_nl = flag_nonlinearity 

        # self.alpha = alpha
        # self.lambda_mu = lambda_mu
        self.myModel = myModel
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = myScaling.get_scaling()
        self.max_iter = max_iter

    # def copos_element_wise(self) :
    #     pass

    def cvx_initialize(self,alpha,Qf=None) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        # optimization variables
        Qcvx = []
        Ycvx = []
        for i in range(N+1) :
            Qcvx.append(cvx.Variable((ix,ix), PSD=True))
            Ycvx.append(cvx.Variable((iu,ix)))
        nu_K = cvx.Variable(N+1)

        # parameters
        A,B = [],[]
        for i in range(N+1) :
            A.append(cvx.Parameter((ix,ix)))
            B.append(cvx.Parameter((ix,iu)))

        Qmax,Rmax = [],[]
        for i in range(N+1) :
            Qmax.append(cvx.Parameter((ix,ix)))
            Rmax.append(cvx.Parameter((iu,iu)))

        # Linear matrix equality
        constraints = []
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            Yi = self.Su@Ycvx[i]@self.Sx
            Ai = A[i]
            Bi = B[i]
            Qip = self.Sx@Qcvx[i+1]@self.Sx # Q_i+1
            Yip = self.Su@Ycvx[i+1]@self.Sx
            Aip = A[i+1]
            Bip = B[i+1]
            
            Fii = Ai@Qi + Bi@Yi + Qi@Ai.T + Yi.T@Bi.T + alpha * Qi
            Fiip = Ai@Qip + Bi@Yip + Qip@Ai.T + Yip.T@Bi.T + alpha * Qip
            Fipi = Aip@Qi + Bip@Yi + Qi@Aip.T + Yi.T@Bip.T + alpha * Qi
            Fipip = Aip@Qip + Bip@Yip + Qip@Aip.T + Yip.T@Bip.T + alpha * Qip

            dQ = (Qip-Qi)/delT
            constraints.append(Fii - dQ << 0)
            constraints.append(Fiip + Fipi - 2*dQ << 0)
            constraints.append(Fipip - dQ << 0)

        # constraints on Q
        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            if Qmax is not None :
                constraints.append(Qi << Qmax[i])
        if Qf is not None :
            Qi = self.Sx@Qcvx[-1]@self.Sx # Q_i
            constraints.append(Qi << Qf)

        # constraints on Y
        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            Yi = self.Su@Ycvx[i]@self.Sx
            tmp1 = cvx.hstack((Qi,Yi.T))
            tmp2 = cvx.hstack((Yi,Rmax[i]))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        for i in range(N+1) :
            Yi = self.Su@Ycvx[i]@self.Sx
            Qi = self.Sx@Qcvx[i]@self.Sx
            tmp1 = cvx.hstack((nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # cost 
        objective_volume = []
        i = 0
        # for i in range(N+1) :
        Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
        objective_volume.append(-cvx.log_det(Qi))
        # objective_volume.append(cvx.trace(Qi))

        
        w_K = 0
        l = cvx.sum(objective_volume) + w_K * cvx.sum(nu_K)

        self.prob = cvx.Problem(cvx.Minimize(l),constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['nu_K'] = nu_K

        # save params
        self.cvx_params = {}
        self.cvx_params['A'] = A
        self.cvx_params['B'] = B
        self.cvx_params['Qmax'] = Qmax
        self.cvx_params['Rmax'] = Rmax

        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l'] = l

    def cvxopt(self,A,B) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        for i in range(N+1) :
            self.cvx_params['A'][i].value = A[i]
            self.cvx_params['B'][i].value = B[i]
            self.cvx_params['Qmax'][i].value = self.Qmax[i]
            self.cvx_params['Rmax'][i].value = self.Rmax[i]

        self.prob.solve(solver=cvx.MOSEK,ignore_dpp=True)
        Qnew = []
        Ynew = []
        for i in range(N+1) :
            Qnew.append(self.Sx@self.cvx_variables['Qcvx'][i].value@self.Sx)
            Ynew.append(self.Su@self.cvx_variables['Ycvx'][i].value@self.Sx)
        Knew = []
        for i in range(N+1) :
            Knew.append(Ynew[i]@np.linalg.inv(Qnew[i]))
        Knew = np.array(Knew)
        Qnew = np.array(Qnew)
        Ynew = np.array(Ynew)
        return Qnew,Knew,Ynew,self.prob.status,self.cvx_cost['l'].value

    def run(self,Qmax,Rmax,xnom,unom) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        self.Qmax = Qmax
        self.Rmax = Rmax

        A,B = self.myModel.diff(xnom,unom)
        assert len(A) == N+1

        # history = []
        self.Qnew,self.Knew,self.Ynew,status,l = self.cvxopt(A,B) 

        self.Q = self.Qnew
        self.K = self.Knew
        self.Y = self.Ynew
        self.c = l

        return self.Q,self.K,self.Y,self.c
