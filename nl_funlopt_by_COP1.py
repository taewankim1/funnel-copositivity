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

class nl_funlopt_by_COP1 :
    def __init__(self,ix,iu,iq,ip,iw,N,delT,myScaling,myModel,max_iter=5,
        w_tr=0,flag_nonlinearity=True) :
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

    def cvx_initialize(self,alpha,lambda_mu,Qini=None,Qf=None,
        Qmax=None,Rmax=None,
        const_state=None,const_input=None) :

        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        # fixed parameters - alpha,lambda_mu

        # optimization variables
        Qcvx = []
        Ycvx = []
        for i in range(N+1) :
            Qcvx.append(cvx.Variable((ix,ix), PSD=True))
            Ycvx.append(cvx.Variable((iu,ix)))
        nu_p = cvx.Variable(N+1,pos=True)
        nu_Q = cvx.Variable(N+1)
        nu_K = cvx.Variable(N+1)
        sv = cvx.Variable(N+1,pos=True) # support value

        # parameters
        A,B,F = [],[],[]
        for i in range(N+1) :
            A.append(cvx.Parameter((ix,ix)))
            B.append(cvx.Parameter((ix,iu)))
            F.append(cvx.Parameter((ix,iw)))
        C = cvx.Parameter((iq,ix))
        D = cvx.Parameter((iq,iu))
        E = cvx.Parameter((ix,ip))
        G = cvx.Parameter((iq,iw))

        gamma_inv_squared = []
        for i in range(N) :
            gamma_inv_squared.append(cvx.Parameter(pos=True))

        # Linear matrix equality
        constraints = []
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx # i
            Yi = self.Su@Ycvx[i]@self.Sx

            Qip = self.Sx@Qcvx[i+1]@self.Sx # i+1
            Yip = self.Su@Ycvx[i+1]@self.Sx

            dQ = (Qip-Qi)/delT
            
            Mii = A[i]@Qi + B[i]@Yi + Qi@A[i].T + Yi.T@B[i].T + (alpha + lambda_mu) * Qi
            Miip = A[i]@Qip + B[i]@Yip + Qip@A[i].T + Yip.T@B[i].T + (alpha+lambda_mu) * Qip
            Mipi = A[i+1]@Qi + B[i+1]@Yi + Qi@A[i+1].T + Yi.T@B[i+1].T + (alpha+lambda_mu) * Qi
            Mipip = A[i+1]@Qip + B[i+1]@Yip + Qip@A[i+1].T + Yip.T@B[i+1].T + (alpha+lambda_mu) * Qip
            Ni = C@Qi + D@Yi
            Nip = C@Qip + D@Yip

            def stack_LMI(LMI11,LMI21,LMI31,LMI41,LMI22,LMI32,LMI42,LMI33,LMI43,LMI44) :
                row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T,LMI41.T))
                row2 = cvx.hstack((LMI21,LMI22,LMI32.T,LMI42.T))
                row3 = cvx.hstack((LMI31,LMI32,LMI33,LMI43.T))
                row4 = cvx.hstack((LMI41,LMI42,LMI43,LMI44))
                LMI = cvx.vstack((row1,row2,row3,row4))
                return LMI


            LMI11 = Mii - dQ
            LMI21 = nu_p[i] * E.T
            LMI31 = F[i].T
            LMI41 = Ni
            LMI22 = -nu_p[i] * np.eye(ip)
            LMI32 = np.zeros((iw,ip))
            LMI42 = np.zeros((iq,ip))
            LMI33 = -lambda_mu * np.eye(iw)
            LMI43 = G
            LMI44 = -nu_p[i] * gamma_inv_squared[i] * np.eye(iq)
            LMI_1 = stack_LMI(LMI11,LMI21,LMI31,LMI41,LMI22,LMI32,LMI42,LMI33,LMI43,LMI44)
            constraints.append(LMI_1 << 0)

            LMI11 = Miip + Mipi - 2*dQ
            LMI21 = (nu_p[i] + nu_p[i+1]) * E.T
            LMI31 = F[i].T + F[i+1].T
            LMI41 = Ni + Nip
            LMI22 = -(nu_p[i]+nu_p[i+1]) * np.eye(ip)
            LMI32 = np.zeros((iw,ip))
            LMI42 = np.zeros((iq,ip))
            LMI33 = -2*lambda_mu * np.eye(iw)
            LMI43 = 2*G
            LMI44 = -(nu_p[i] + nu_p[i+1]) * gamma_inv_squared[i] * np.eye(iq)
            LMI_2 = stack_LMI(LMI11,LMI21,LMI31,LMI41,LMI22,LMI32,LMI42,LMI33,LMI43,LMI44)
            constraints.append(LMI_2 << 0)

            LMI11 = Mipip - dQ
            LMI21 = nu_p[i+1] * E.T
            LMI31 = F[i+1].T
            LMI41 = Nip
            LMI22 = -nu_p[i+1] * np.eye(ip)
            LMI32 = np.zeros((iw,ip))
            LMI42 = np.zeros((iq,ip))
            LMI33 = -lambda_mu * np.eye(iw)
            LMI43 = G
            LMI44 = -nu_p[i+1] * gamma_inv_squared[i] * np.eye(iq)
            LMI_3 = stack_LMI(LMI11,LMI21,LMI31,LMI41,LMI22,LMI32,LMI42,LMI33,LMI43,LMI44)
            constraints.append(LMI_3 << 0)

        # constraints on sv
        for i in range(N+1) :
            constraints.append(sv[i] <= 1)
            if i > 0 :
                constraints.append(sv[i] * np.exp(-alpha*delT*i) <= sv[0])

        # constraints on Q
        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            constraints.append(Qi << nu_Q[i]*np.eye(ix))
            if Qmax is not None :
                constraints.append(Qi << sv[i]*Qmax[i])

        if const_state is not None :
            for const in const_state :
                for i in range(N+1) :
                    Qi = self.Sx@Qcvx[i]@self.Sx 
                    tmp = (sv[i] * const['(b-ax)^2'][i])[np.newaxis,np.newaxis]
                    tmp1 = cvx.hstack((tmp,const['a'][i].T@Qi))
                    tmp2 = cvx.hstack((Qi.T@const['a'][i],Qi))
                    constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # constraints on Y
        if const_input is not None :
            for const in const_input :
                for i in range(N+1) :
                    Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
                    Yi = self.Su@Ycvx[i]@self.Sx
                    tmp = (sv[i] * const['(b-au)^2'][i])[np.newaxis,np.newaxis]
                    tmp1 = cvx.hstack((tmp,const['a'][i].T@Yi))
                    tmp2 = cvx.hstack((Yi.T@const['a'][i],Qi))
                    constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        if Rmax is not None :
            for i in range(N+1) :
                Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
                Yi = self.Su@Ycvx[i]@self.Sx
                tmp1 = cvx.hstack((Qi,Yi.T))
                tmp2 = cvx.hstack((Yi,sv[i]*Rmax[i]))
                constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        for i in range(N+1) :
            Yi = self.Su@Ycvx[i]@self.Sx
            Qi = self.Sx@Qcvx[i]@self.Sx
            tmp1 = cvx.hstack((nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # boundary condition
        if Qini is not None :
            Qi = self.Sx@Qcvx[0]@self.Sx    
            constraints.append(Qi >> sv[0]*Qini)
        if Qf is not None :
            Qi = self.Sx@Qcvx[-1]@self.Sx # Q_i
            constraints.append(Qi << sv[-1]*Qf)

        Q0 = self.Sx@Qcvx[0]@self.Sx # Q_i
        l = 1e3*sv[0] + 1e-1*(-cvx.log_det(Q0)) + 1e-1*cvx.sum(nu_Q) + 0*cvx.sum(nu_K)

        self.prob = cvx.Problem(cvx.Minimize(l),constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['nu_p'] = nu_p
        self.cvx_variables['nu_Q'] = nu_Q
        self.cvx_variables['nu_K'] = nu_K
        self.cvx_variables['sv'] = sv

        # save params
        self.cvx_params = {}
        self.cvx_params['A'] = A
        self.cvx_params['B'] = B
        self.cvx_params['C'] = C
        self.cvx_params['D'] = D
        self.cvx_params['E'] = E
        self.cvx_params['F'] = F
        self.cvx_params['G'] = G
        self.cvx_params['gamma_inv_squared'] = gamma_inv_squared

        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l'] = l

    def cvxopt(self,gamma,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        for i in range(N+1) :
            self.cvx_params['A'][i].value = A[i]
            self.cvx_params['B'][i].value = B[i]
            self.cvx_params['F'][i].value = F[i]
            if i < N :
                self.cvx_params['gamma_inv_squared'][i].value = 1 / (gamma[i]**2)
        self.cvx_params['C'].value = C
        self.cvx_params['D'].value = D
        self.cvx_params['E'].value = E
        self.cvx_params['G'].value = G

        self.prob.solve(solver=cvx.MOSEK,ignore_dpp=True)
        print(self.prob.status)
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
        svnew = self.cvx_variables['sv'].value
        return Qnew,Knew,Ynew,svnew,self.prob.status,self.cvx_cost['l'].value

    def run(self,gamma,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        # A,B = self.myModel.diff(xnom,unom)
        # assert len(A) == N+1

        # history = []
        self.Qnew,self.Knew,self.Ynew,sv,status,l = self.cvxopt(gamma,A,B,C,D,E,F,G) 

        self.Q = self.Qnew
        self.K = self.Knew
        self.Y = self.Ynew
        self.sv = sv
        self.c = l

        return self.Q,self.K,self.Y,self.sv,self.c
