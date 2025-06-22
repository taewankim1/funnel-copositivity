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

class funlopt_by_OCP_refine :
    def __init__(self,ix,iu,N,delT,myLMI,myScaling,myModel,max_iter=5,
        w_tr=1,flag_nonlinearity=True) :
        self.ix = ix
        self.iu = iu
        self.delT = delT
        self.N = N
        self.small = 1e-8
        self.w_tr = w_tr
        self.flag_nl = flag_nonlinearity 

        self.myModel = myModel
        self.myLMI = myLMI
        self.Sx,_,_,_,_,_ = myScaling.get_scaling()
        self.max_iter = max_iter


    def cvx_initialize(self,Pmax,Pf=None) :
        ix,iu,N = self.ix,self.iu,self.N

        # optimization variables
        Pcvx = []
        Zcvx = []
        for i in range(N+1) :
            Pcvx.append(cvx.Variable((ix,ix), PSD=True))
            Zcvx.append(cvx.Variable((ix,ix), PSD=True))

        # parameters
        Ap,Sm,Sp = [],[],[]
        for i in range(N) :
            Ap.append(cvx.Parameter((ix*ix,ix*ix)))
            Sm.append(cvx.Parameter((ix*ix,ix*ix)))
            Sp.append(cvx.Parameter((ix*ix,ix*ix)))

        Qinv,K = [],[]
        for i in range(N+1) :
            Qinv.append(cvx.Parameter((ix,ix),PSD=True))
            K.append(cvx.Parameter((iu,ix)))

        Pmin,Rmax = [],[]
        for i in range(N+1) :
            Pmin.append(cvx.Parameter((ix,ix)))
            Rmax.append(cvx.Parameter((iu,iu)))

        constraints = []
        # Linear matrix equality
        for i in range(N) :
            Pi = self.Sx@Pcvx[i]@self.Sx # Q_i
            Zi = self.Sx@Zcvx[i]@self.Sx
            Pip = self.Sx@Pcvx[i+1]@self.Sx # Q_i+1
            Zip = self.Sx@Zcvx[i+1]@self.Sx

            constraints.append(vec(Pip) == Ap[i]@vec(Pi)
                + Sm[i]@vec(Zi) + Sp[i]@vec(Zip)
                )

        # constraints on P
        for i in range(N+1) :
            Pi = self.Sx@Pcvx[i]@self.Sx # Q_i
            constraints.append(Pi >> Pmin[i]) # PD
            constraints.append(Pi << Pmax)
        if Pf is not None :
            Pi = self.Sx@Pcvx[-1]@self.Sx # Q_i
            constraints.append(Pi >> Pf)
            

        # constraints on K
        for i in range(N+1) :
            Pi = self.Sx@Pcvx[i]@self.Sx # Q_i
            Ki = K[i]
            tmp1 = cvx.hstack((Pi,Ki.T))
            tmp2 = cvx.hstack((Ki,Rmax[i]))
            # constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)


        # cost 
        objective = []
        # i = 0
        for i in range(N+1) :
            Pi = self.Sx@Pcvx[i]@self.Sx # Q_i
            objective.append(cvx.norm(Pi - Qinv[i]))

        l = cvx.sum(objective)


        self.prob = cvx.Problem(cvx.Minimize(l),constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Pcvx'] = Pcvx
        self.cvx_variables['Zcvx'] = Zcvx

        # save params
        self.cvx_params = {}
        self.cvx_params['Ap'] = Ap
        self.cvx_params['Sm'] = Sm
        self.cvx_params['Sp'] = Sp
        self.cvx_params['Pmin'] = Pmin
        self.cvx_params['Rmax'] = Rmax
        self.cvx_params['Qinv'] = Qinv
        self.cvx_params['K'] = K

        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l'] = l

    def cvxopt(self) :
        ix,iu,N = self.ix,self.iu,self.N

        for i in range(N+1) :
            if i < N :
                self.cvx_params['Ap'][i].value = self.Ap[i]
                self.cvx_params['Sm'][i].value= self.Sm[i]
                self.cvx_params['Sp'][i].value = self.Sp[i]
            self.cvx_params['Pmin'][i].value = self.Pmin[i]
            self.cvx_params['Rmax'][i].value = self.Rmax[i]
            self.cvx_params['Qinv'][i].value = self.Qinv[i]
            self.cvx_params['K'][i].value = self.K[i]

        self.prob.solve(solver=cvx.MOSEK,ignore_dpp=True)
        Pnew = []
        Znew = []
        for i in range(N+1) :
            Pnew.append(self.Sx@self.cvx_variables['Pcvx'][i].value@self.Sx)
            Znew.append(self.Sx@self.cvx_variables['Zcvx'][i].value@self.Sx)
        Pnew = np.array(Pnew)
        return Pnew,Znew,self.prob.status,self.cvx_cost['l'].value

    def run(self,Qinv,K,Z0,Pmin,Rmax,xnom,unom,flag_FOH=False) :
        ix,iu,N = self.ix,self.iu,self.N
        delT = self.delT

        self.P = Qinv
        self.Qinv = Qinv
        self.K = K
        self.Z = Z0

        self.Pmin = Pmin
        self.Rmax = Rmax

        self.c = 0

        if flag_FOH is False :
            Anom,Bnom = None,None
        else :
            Anom,Bnom = self.myModel.diff(xnom,unom)

        history = []

        # iteration starts
        for iteration in range(self.max_iter) :
            history_iter = {}
            # step 1. differentiate dynamics
            self.Ap,self.Sm,self.Sp,x_prop,_ = self.myLMI.discrete_foh(
                xnom,
                unom,
                self.P,self.K,self.Z,
                delT,self.myModel,
                Anom,Bnom
                )
            eps_machine = np.finfo(float).eps
            self.Ap[np.abs(self.Ap) < eps_machine] = 0
            self.Sm[np.abs(self.Sm) < eps_machine] = 0
            self.Sp[np.abs(self.Sp) < eps_machine] = 0

            # step2. cvxopt
            self.Pnew,self.Znew,status,l = self.cvxopt() 

            # step3. evaluation
            # write a code!
            self.P = self.Pnew
            self.Z = self.Znew
            self.c = l

        return self.P,self.Z,self.c,x_prop




