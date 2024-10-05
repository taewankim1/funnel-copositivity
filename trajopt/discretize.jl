include("dynamics.jl")
function print_jl(x)
    println("Type is $(typeof(x))")
    println("Shape is $(size(x))")
end
function RK4(ODEFun::Function, y0::Vector, tspan::Tuple, p::Any,  N_RK::Int64)
    t = collect(range(tspan[1], stop=tspan[2], length=N_RK+1))
    h = t[2] - t[1]
    iy = size(y0,1)
    ysol = zeros(iy,N_RK+1)
    ysol[:,1] .= y0
    k1 = zeros(size(y0))
    k2 = zeros(size(y0))
    k3 = zeros(size(y0))
    k4 = zeros(size(y0))
    for i = 1:N_RK
        tk = t[i]
        yk = ysol[:,i];
        ODEFun(k1,yk,p,tk)
        ODEFun(k2,yk+ h/2*k1,p,tk + h/2)
        ODEFun(k3,yk+ h/2*k2,p,tk + h/2)
        ODEFun(k4,yk+ h*k3,p,tk + h)
        ysol[:,i+1] = yk + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    end
    return t, ysol
end
"""
    diff_numeric_central(fun,x,u)

get fx, fu using finite difference

# Arguments
- 
# Returns
-
"""
function diff(model::Dynamics,x::Vector,u::Vector)
    ix = length(x)
    iu = length(u)
    eps_x = Diagonal{Float64}(I, ix)
    eps_u = Diagonal{Float64}(I, iu)
    fx = zeros(ix,ix)
    fu = zeros(ix,iu)
    h = 2^(-18)
    for i in 1:ix
        fx[:,i] = (forward(model,x+h*eps_x[:,i],u) - forward(model,x-h*eps_x[:,i],u)) / (2*h)
    end
    for i in 1:iu
        fu[:,i] = (forward(model,x,u+h*eps_u[:,i]) - forward(model,x,u-h*eps_u[:,i])) / (2*h)
    end
    return fx,fu
end
"""
    discretize_foh(fun,x,u)

variational method with FOH control parameterization

# Arguments
- 
# Returns
-
"""
function discretize_foh(model::Dynamics,x::Matrix,u::Matrix,T::Vector)
    N = size(x,2)
    ix = size(x,1)
    iy = size(x,1)

    idx_state = 1:ix
    idx_A = (ix+1):(ix+ix*ix)
    idx_Bm = (ix+ix*ix+1):(ix+ix*ix+ix*iu)
    idx_Bp = (ix+ix*ix+ix*iu+1):(ix+ix*ix+2*ix*iu)
    idx_s = (ix+ix*ix+2*ix*iu+1):(ix+ix*ix+2*ix*iu+ix)
    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        dt = p[3]
        alpha = 1 - t
        beta = t
        u1 = alpha * um + beta * up
        x1 = V[idx_state]
        Phi = reshape(V[idx_A], (ix, ix))
        x3 = reshape(V[idx_Bm],(ix,iu))
        x4 = reshape(V[idx_Bp],(ix,iu))
        x5 = reshape(V[idx_s],(ix,1))
        f = forward(model,x1,u1)
        dA,dB = diff(model,x1,u1)
        dA = dA*dt
        dB = dB*dt
        dpdt = dA*Phi
        dbmdt = dA*x3 + dB*alpha
        dbpdt = dA*x4 + dB*beta
        dsdt = dA*x5 + f
        dv = [dt*f;dpdt[:];dbmdt[:];dbpdt[:];dsdt[:]]
        out .= dv[:]
    end
    A = zeros(ix,ix,N)
    Bm = zeros(ix,iu,N)
    Bp = zeros(ix,iu,N)
    s = zeros(ix,N)
    z = zeros(ix,N)
    x_prop = zeros(ix,N)
    for i = 1:N
        A0 = Matrix{Float64}(I, ix, ix)
        Bm0 = zeros(ix,iu)
        Bp0 = zeros(ix,iu)
        s0 = zeros(ix,1)
        V0 = [x[:,i];A0[:];Bm0[:];Bp0[:];s0][:]
        um = u[:,i]
        up = u[:,i+1]
        dt = T[i]
        t, sol = RK4(dvdt,V0,(0,1),(um,up,dt),50)
        x_prop[:,i] .= sol[idx_state,end]
        A[:,:,i] .= reshape(sol[idx_A,end],ix,ix)
        Bm[:,:,i] .= reshape(sol[idx_Bm,end],ix,iu)
        Bp[:,:,i] .= reshape(sol[idx_Bp,end],ix,iu)
        s[:,i] .= sol[idx_s,end]
        z[:,i] .= x_prop[:,i] - A[:,:,i]*x[:,i] - Bm[:,:,i]*um - Bp[:,:,i]*up - s[:,i] * dt
    end
    return A,Bm,Bp,s,z,x_prop
end