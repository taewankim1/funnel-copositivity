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

class funlopt_by_OCP :
    def __init__(self,ix,iu,iq,ip,iw,N,delT,myLMI,myScaling,myModel,max_iter=5,
        alpha=0.25,lambda_mu=0.1,
        w_tr=1,flag_nonlinearity=True) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.delT = delT
        self.N = N
        self.small = 1e-8
        self.w_tr = w_tr
        self.flag_nl = flag_nonlinearity 

        # self.alpha = alpha
        # self.lambda_mu = lambda_mu

        self.myModel = myModel
        self.myLMI = myLMI
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = myScaling.get_scaling()
        self.max_iter = max_iter


    def cvx_initialize(self,Qf=None) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        # optimization variables
        Qcvx = []
        Ycvx = []
        Zcvx = []
        for i in range(N+1) :
            Qcvx.append(cvx.Variable((ix,ix), PSD=True))
            Ycvx.append(cvx.Variable((iu,ix)))
            Zcvx.append(cvx.Variable((ix,ix), PSD=True))
        nu_K = cvx.Variable(N+1)

        # parameters
        Aq,Bm,Bp,Sm,Sp = [],[],[],[],[]
        for i in range(N) :
            Aq.append(cvx.Parameter((ix*ix,ix*ix)))
            Bm.append(cvx.Parameter((ix*ix,iu*ix)))
            Bp.append(cvx.Parameter((ix*ix,iu*ix)))
            Sm.append(cvx.Parameter((ix*ix,ix*ix)))
            Sp.append(cvx.Parameter((ix*ix,ix*ix)))

        Qmax,Rmax = [],[]
        for i in range(N+1) :
            Qmax.append(cvx.Parameter((ix,ix)))
            Rmax.append(cvx.Parameter((iu,iu)))

        constraints = []
        # Linear matrix equality
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            Yi = self.Su@Ycvx[i]@self.Sx
            Zi = self.Sx@Zcvx[i]@self.Sx
            Qip = self.Sx@Qcvx[i+1]@self.Sx # Q_i+1
            Yip = self.Su@Ycvx[i+1]@self.Sx
            Zip = self.Sx@Zcvx[i+1]@self.Sx

            constraints.append(vec(Qip) == Aq[i]@vec(Qi)
                + Bm[i]@vec(Yi) + Bp[i]@vec(Yip)
                + Sm[i]@vec(Zi) + Sp[i]@vec(Zip)
                )

        # # constraints on Z :
        # for i in range(N+1) :
        #     constraints.append(Zcvx[i] == 0)
        i = 0
        # # # for i in range(N+1) :
        Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
        constraints.append(Qi << 0.001*np.eye(ix))

        # constraints on Q
        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx # Q_i
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
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
        self.cvx_variables['Zcvx'] = Zcvx
        self.cvx_variables['nu_K'] = nu_K

        # save params
        self.cvx_params = {}
        self.cvx_params['Aq'] = Aq
        self.cvx_params['Bm'] = Bm
        self.cvx_params['Bp'] = Bp
        self.cvx_params['Sm'] = Sm
        self.cvx_params['Sp'] = Sp
        self.cvx_params['Qmax'] = Qmax
        self.cvx_params['Rmax'] = Rmax

        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l'] = l

    def cvxopt(self) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        for i in range(N) :
            self.cvx_params['Aq'][i].value = self.Aq[i]
            self.cvx_params['Bm'][i].value = self.Bm[i]
            self.cvx_params['Bp'][i].value = self.Bp[i]
            self.cvx_params['Sm'][i].value= self.Sm[i]
            self.cvx_params['Sp'][i].value = self.Sp[i]
            self.cvx_params['Qmax'][i].value = self.Qmax[i]
            self.cvx_params['Rmax'][i].value = self.Rmax[i]
        self.cvx_params['Qmax'][-1].value = self.Qmax[-1]
        self.cvx_params['Rmax'][-1].value = self.Rmax[-1]

        self.prob.solve(solver=cvx.MOSEK,ignore_dpp=True)
        Qnew = []
        Znew = []
        Ynew = []
        for i in range(N+1) :
            Qnew.append(self.Sx@self.cvx_variables['Qcvx'][i].value@self.Sx)
            Ynew.append(self.Su@self.cvx_variables['Ycvx'][i].value@self.Sx)
            Znew.append(self.Sx@self.cvx_variables['Zcvx'][i].value@self.Sx)
        Knew = []
        for i in range(N+1) :
            Knew.append(Ynew[i]@np.linalg.inv(Qnew[i]))
        Knew = np.array(Knew)
        Qnew = np.array(Qnew)
        Ynew = np.array(Ynew)
        return Qnew,Knew,Ynew,Znew,self.prob.status,self.cvx_cost['l'].value

    def run(self,Q0,Y0,Z0,Qmax,Rmax,xnom,unom,flag_FOH=False) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        delT = self.delT

        self.Q = Q0
        self.Y = Y0
        self.Z = Z0

        self.Qmax = Qmax
        self.Rmax = Rmax

        if flag_FOH is False :
            Anom,Bnom = None,None
        else :
            Anom,Bnom = self.myModel.diff(xnom,unom)

        history = []

        # iteration starts
        for iteration in range(self.max_iter) :
            history_iter = {}
            # step 1. differentiate dynamics
            self.Aq,self.Bm,self.Bp,self.Sm,self.Sp,x_prop,_ = self.myLMI.discrete_foh(
                xnom,
                unom,
                self.Q,self.Y,self.Z,
                delT,self.myModel,
                Anom,Bnom
                )
            eps_machine = np.finfo(float).eps
            self.Aq[np.abs(self.Aq) < eps_machine] = 0
            self.Bm[np.abs(self.Bm) < eps_machine] = 0
            self.Bp[np.abs(self.Bp) < eps_machine] = 0
            self.Sm[np.abs(self.Sm) < eps_machine] = 0
            self.Sp[np.abs(self.Sp) < eps_machine] = 0


            # step2. cvxopt
            self.Qnew,self.Knew,self.Ynew,self.Znew,status,l = self.cvxopt() 

            # step3. evaluation
            # write a code!
            self.Q = self.Qnew
            self.K = self.Knew
            self.Y = self.Ynew
            self.Z = self.Znew
            self.c = l

        return self.Q,self.K,self.Y,self.Z,self.c,x_prop




