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

class nl_funlopt_by_COP :
    def __init__(self,ix,iu,iq,iphi,iw,N,delT,myScaling,myModel,max_iter=5,
        w_tr=0,flag_nonlinearity=True,type_copositivity=1) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.iphi = iphi
        self.iw = iw
        self.ir = ix+iu+iw+iq
        self.ip = ix + iphi
        self.delT = delT
        self.N = N
        self.small = 1e-6
        self.w_tr = w_tr
        self.flag_nl = flag_nonlinearity 
        self.myModel = myModel
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = myScaling.get_scaling()
        self.max_iter = max_iter
        self.type_copositivity = type_copositivity

    def cvx_initialize(self,lambda_w,Qini=None,Qf=None,
        Qmax=None,Rmax=None,
        const_state=None,const_input=None) :

        ix,iu,N = self.ix,self.iu,self.N
        iq,iphi,iw = self.iq,self.iphi,self.iw
        ip,ir = self.ip,self.ir
        delT = self.delT

        # optimization variables
        Qcvx = []
        Ycvx = []
        for i in range(N+1) :
            Qcvx.append(cvx.Variable((ix,ix), PSD=True))
            Ycvx.append(cvx.Variable((iu,ix)))
        # lam_gamma = cvx.Variable(N+1,pos=True)
        # lam_beta = cvx.Variable(N+1,pos=True)
        lam = cvx.Variable((N+1,2),pos=True) # for beta, gamma
        # nu_Q = cvx.Variable(N+1)

        # parameters
        A,B,F = [],[],[]
        for i in range(N+1) :
            A.append(cvx.Parameter((ix,ix)))
            B.append(cvx.Parameter((ix,iu)))
            F.append(cvx.Parameter((ix,iw)))
        C = cvx.Parameter((ir,ix))
        D = cvx.Parameter((ir,iu))
        E = cvx.Parameter((ix,ip))
        G = cvx.Parameter((ir,iw))

        gamma_inv_squared = []
        beta_inv_squared = []
        for i in range(N) :
            gamma_inv_squared.append(cvx.Parameter(pos=True))
            beta_inv_squared.append(cvx.Parameter(pos=True))
            
        def stack_LMI(LMI11,LMI21,LMI31,LMI41,
                            LMI22,LMI32,LMI42,
                                  LMI33,LMI43,
                                        LMI44) :
            row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T,LMI41.T))
            row2 = cvx.hstack((LMI21,LMI22,LMI32.T,LMI42.T))
            row3 = cvx.hstack((LMI31,LMI32,LMI33,LMI43.T))
            row4 = cvx.hstack((LMI41,LMI42,LMI43,LMI44))
            LMI = cvx.vstack((row1,row2,row3,row4))
            return LMI

        def get_N1(i,gamma_inv,beta_inv) :
            N1 = cvx.hstack((lam[i,0]*beta_inv*np.eye(ix+iu+iw),np.zeros((ix+iu+iw,iq))))
            N2 = cvx.hstack((np.zeros((iq,ix+iu+iw)),lam[i,1]*gamma_inv*np.eye(iq)))
            N = cvx.vstack((N1,N2))
            return N

        def get_N2(i) :
            N1 = cvx.hstack((lam[i,0]*np.eye(ix),np.zeros((ix,iphi))))
            N2 = cvx.hstack((np.zeros((iphi,ix)),lam[i,1]*np.eye(iphi)))
            N = cvx.vstack((N1,N2))
            return N

        def get_H(i,j,dQ,gamma_inv,beta_inv) :
            Qi = self.Sx@Qcvx[i]@self.Sx # i
            Yi = self.Su@Ycvx[i]@self.Sx

            Qj = self.Sx@Qcvx[j]@self.Sx # j
            Yj = self.Su@Ycvx[j]@self.Sx

            Wij = A[i]@Qj + B[i]@Yj + Qj@A[i].T + Yj.T@B[i].T + lambda_w * Qj
            Wji = A[j]@Qi + B[j]@Yi + Qi@A[j].T + Yi.T@B[j].T + lambda_w * Qi

            Li = C@Qi + D@Yi
            Lj = C@Qj + D@Yj
            
            LMI11 = Wij + Wji - 2*dQ
            LMI21 = (get_N2(i) + get_N2(j)) @ E.T
            LMI31 = F[i].T + F[j].T
            LMI41 = Li + Lj
            LMI22 = - (get_N2(i)+get_N2(j))
            LMI32 = np.zeros((iw,ip))
            LMI42 = np.zeros((ir,ip))
            LMI33 = -2*lambda_w * np.eye(iw)
            LMI43 = 2*G
            LMI44 = - (get_N1(i,gamma_inv,beta_inv) + get_N1(j,gamma_inv,beta_inv))
            H_ij = - 1/2 * stack_LMI(LMI11,LMI21,LMI31,LMI41,
                                           LMI22,LMI32,LMI42,
                                                 LMI33,LMI43,
                                                       LMI44)
            return H_ij

        def copositive_to_LMI_1(constraints) :
            for i in range(N) :
                j = i+1
                Qi = self.Sx@Qcvx[i]@self.Sx # i
                Qj = self.Sx@Qcvx[j]@self.Sx # j
                dQ = (Qj-Qi)/delT
                
                gamma_inv = gamma_inv_squared[i]
                beta_inv = beta_inv_squared[i]

                H_ii = get_H(i,i,dQ,gamma_inv,beta_inv)
                constraints.append(H_ii >> 0)

                H_ij_ji = get_H(i,j,dQ,gamma_inv,beta_inv)
                constraints.append(H_ij_ji >> 0)

                H_jj = get_H(j,j,dQ,gamma_inv,beta_inv)
                constraints.append(H_jj >> 0)
            return
        def copositive_to_LMI_2(constraints) :
            X11 = []
            X21 = []
            X22 = []
            iH = ix+ip+iw+ir
            for i in range(N) :
                X11.append(cvx.Variable((iH,iH), PSD=True))
                X21.append(cvx.Variable((iH,iH), PSD=True))
                X22.append(cvx.Variable((iH,iH), PSD=True))
            for i in range(N) :
                j = i+1
                Qi = self.Sx@Qcvx[i]@self.Sx # i
                Qj = self.Sx@Qcvx[j]@self.Sx # j
                dQ = (Qj-Qi)/delT
                gamma_inv = gamma_inv_squared[i]
                beta_inv = beta_inv_squared[i]

                H_ii = get_H(i,i,dQ,gamma_inv,beta_inv)
                H_ij_ji = get_H(i,j,dQ,gamma_inv,beta_inv)
                H_jj = get_H(j,j,dQ,gamma_inv,beta_inv)

                LMI11 = H_ii - X11[i]
                LMI21 = H_ij_ji - X21[i]
                LMI22 = H_jj - X22[i]

                row1 = cvx.hstack((LMI11,LMI21.T))
                row2 = cvx.hstack((LMI21,LMI22))
                LMI = cvx.vstack((row1,row2))
                constraints.append(LMI >> 0)
            return

        # Linear matrix equality
        constraints = []
        if self.type_copositivity == 1 :
            copositive_to_LMI_1(constraints)
        elif self.type_copositivity == 2:
            copositive_to_LMI_2(constraints)
        else :
            raise TypeError("Value must be 1 or 2")

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
        QN = self.Sx@Qcvx[-1]@self.Sx # Q_i
        l = -cvx.trace(Q0) + cvx.trace(QN)
        # l = -cvx.log_det(Q0) + cvx.trace(QN)
        # l = -cvx.lambda_min(Q0) + cvx.lambda_max(QN)
        # l = -cvx.trace(Q0)
        # l = cvx.trace(Q0)

        self.prob = cvx.Problem(cvx.Minimize(l),constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['lam'] = lam
        # self.cvx_variables['nu_p'] = nu_p
        # self.cvx_variables['lam_beta'] = lam_beta
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
        self.cvx_params['beta_inv_squared'] = beta_inv_squared

        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l'] = l

    def cvxopt(self,gamma,beta,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        for i in range(N+1) :
            self.cvx_params['A'][i].value = A[i]
            self.cvx_params['B'][i].value = B[i]
            self.cvx_params['F'][i].value = F[i]
            if i < N :
                self.cvx_params['gamma_inv_squared'][i].value = 1 / (gamma[i]**2)
                self.cvx_params['beta_inv_squared'][i].value = 1 / (beta[i]**2)
        self.cvx_params['C'].value = C
        self.cvx_params['D'].value = D
        self.cvx_params['E'].value = E
        self.cvx_params['G'].value = G

        # self.prob.solve(solver=cvx.CLARABEL)
        self.prob.solve(solver=cvx.MOSEK)
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
        # lam_gamma_new = self.cvx_variables['lam_gamma'].value
        # lam_beta_new = self.cvx_variables['lam_beta'].value
        return Qnew,Knew,Ynew,self.prob.status,self.cvx_cost['l'].value

    def run(self,gamma,beta,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        # A,B = self.myModel.diff(xnom,unom)
        # assert len(A) == N+1

        # history = []
        self.Qnew,self.Knew,self.Ynew,status,l = self.cvxopt(gamma,beta,A,B,C,D,E,F,G) 

        self.Q = self.Qnew
        self.K = self.Knew
        self.Y = self.Ynew
        self.c = l

        return self.Q,self.K,self.Y,self.c
