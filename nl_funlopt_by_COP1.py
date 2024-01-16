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
        self.myModel = myModel
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = myScaling.get_scaling()
        self.max_iter = max_iter

    # def copos_element_wise(self) :
    #     pass

    def cvx_initialize(self,lambda_w,Qini=None,Qf=None,
        Qmax=None,Rmax=None,
        const_state=None,const_input=None) :

        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        # optimization variables
        Qcvx = []
        Ycvx = []
        for i in range(N+1) :
            Qcvx.append(cvx.Variable((ix,ix), PSD=True))
            Ycvx.append(cvx.Variable((iu,ix)))
        nu_p = cvx.Variable(N+1,pos=True)
        # nu_Q = cvx.Variable(N+1)
        # nu_K = cvx.Variable(N+1)

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
            
        def stack_LMI(LMI11,LMI21,LMI31,LMI41,LMI22,LMI32,LMI42,LMI33,LMI43,LMI44) :
            row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T,LMI41.T))
            row2 = cvx.hstack((LMI21,LMI22,LMI32.T,LMI42.T))
            row3 = cvx.hstack((LMI31,LMI32,LMI33,LMI43.T))
            row4 = cvx.hstack((LMI41,LMI42,LMI43,LMI44))
            LMI = cvx.vstack((row1,row2,row3,row4))
            return LMI

        def get_H(i,j,dQ,gamma_inv) :
            Qi = self.Sx@Qcvx[i]@self.Sx # i
            Yi = self.Su@Ycvx[i]@self.Sx

            Qj = self.Sx@Qcvx[j]@self.Sx # j
            Yj = self.Su@Ycvx[j]@self.Sx

            # dQ = (Qip-Qi)/delT
            Wij = A[i]@Qj + B[i]@Yj + Qj@A[i].T + Yj.T@B[i].T + lambda_w * Qj
            Wji = A[j]@Qi + B[j]@Yi + Qi@A[j].T + Yi.T@B[j].T + lambda_w * Qi

            Li = C@Qi + D@Yi
            Lj = C@Qj + D@Yj

            LMI11 = Wij + Wji - 2*dQ
            LMI21 = (nu_p[i] + nu_p[j]) * E.T
            LMI31 = F[i].T + F[j].T
            LMI41 = Li + Lj
            LMI22 = -(nu_p[i]+nu_p[j]) * np.eye(ip)
            LMI32 = np.zeros((iw,ip))
            LMI42 = np.zeros((iq,ip))
            LMI33 = -2*lambda_w * np.eye(iw)
            LMI43 = 2*G
            # LMI44 = -(nu_p[i] + nu_p[j]) * gamma_inv_squared[i] * np.eye(iq)
            LMI44 = -(nu_p[i] + nu_p[j]) * gamma_inv * np.eye(iq)
            H_ij = - 1/2 * stack_LMI(LMI11,LMI21,LMI31,LMI41,LMI22,LMI32,LMI42,LMI33,LMI43,LMI44)
            return H_ij

        # Linear matrix equality
        constraints = []
        for i in range(N) :
            j = i+1
            Qi = self.Sx@Qcvx[i]@self.Sx # i
            Qj = self.Sx@Qcvx[j]@self.Sx # i+1
            dQ = (Qj-Qi)/delT
            gamma_inv = gamma_inv_squared[i]

            H_ii = get_H(i,i,dQ,gamma_inv)
            constraints.append(H_ii >> 0)

            H_ij_ji = get_H(i,j,dQ,gamma_inv)
            constraints.append(H_ij_ji >> 0)

            H_jj = get_H(j,j,dQ,gamma_inv)
            constraints.append(H_jj >> 0)

        # constraints on Q
        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            # constraints.append(Qi << nu_Q[i]*np.eye(ix))
            if Qmax is not None :
                constraints.append(Qi << Qmax[i])
        if const_state is not None :
            for const in const_state :
                for i in range(N+1) :
                    Qi = self.Sx@Qcvx[i]@self.Sx 
                    tmp = (const['(b-ax)^2'][i])[np.newaxis,np.newaxis]
                    tmp1 = cvx.hstack((tmp,const['a'][i].T@Qi))
                    tmp2 = cvx.hstack((Qi.T@const['a'][i],Qi))
                    constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        # constraints on Y
        if const_input is not None :
            for const in const_input :
                for i in range(N+1) :
                    Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
                    Yi = self.Su@Ycvx[i]@self.Sx
                    tmp = (const['(b-au)^2'][i])[np.newaxis,np.newaxis]
                    tmp1 = cvx.hstack((tmp,const['a'][i].T@Yi))
                    tmp2 = cvx.hstack((Yi.T@const['a'][i],Qi))
                    constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        if Rmax is not None :
            for i in range(N+1) :
                Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
                Yi = self.Su@Ycvx[i]@self.Sx
                tmp1 = cvx.hstack((Qi,Yi.T))
                tmp2 = cvx.hstack((Yi,Rmax[i]))
                constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        # for i in range(N+1) :
        #     Yi = self.Su@Ycvx[i]@self.Sx
        #     Qi = self.Sx@Qcvx[i]@self.Sx
        #     tmp1 = cvx.hstack((nu_K[i]*np.eye(iu),Yi))
        #     tmp2 = cvx.hstack((Yi.T,Qi))
        #     constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # boundary condition
        if Qini is not None :
            Qi = self.Sx@Qcvx[0]@self.Sx    
            constraints.append(Qi >> Qini)
        if Qf is not None :
            Qi = self.Sx@Qcvx[-1]@self.Sx # Q_i
            constraints.append(Qi << Qf)

        Q0 = self.Sx@Qcvx[0]@self.Sx # Q_i
        # l = 1e3*sv[0] + 1e-1*(-cvx.log_det(Q0)) + 1e-1*cvx.sum(nu_Q) + 0*cvx.sum(nu_K)
        l = -cvx.log_det(Q0)

        self.prob = cvx.Problem(cvx.Minimize(l),constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['nu_p'] = nu_p
        # self.cvx_variables['nu_Q'] = nu_Q
        # self.cvx_variables['nu_K'] = nu_K
        # self.cvx_variables['sv'] = sv

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

        self.prob.solve(solver=cvx.CLARABEL,ignore_dpp=True)
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
        # svnew = self.cvx_variables['sv'].value
        return Qnew,Knew,Ynew,self.prob.status,self.cvx_cost['l'].value

    def run(self,gamma,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        # A,B = self.myModel.diff(xnom,unom)
        # assert len(A) == N+1

        # history = []
        self.Qnew,self.Knew,self.Ynew,status,l = self.cvxopt(gamma,A,B,C,D,E,F,G) 

        self.Q = self.Qnew
        self.K = self.Knew
        self.Y = self.Ynew
        self.c = l

        return self.Q,self.K,self.Y,self.c
