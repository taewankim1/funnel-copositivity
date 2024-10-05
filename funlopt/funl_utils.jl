
include("../trajopt/dynamics.jl")
include("../trajopt/discretize.jl")
include("funl_dynamics.jl")
using LinearAlgebra

function get_radius_angle_Ellipse2D(Q_list)
    radius_list = []
    angle_list = []

    for i in 1:size(Q_list,3)
        Q_ = Q_list[:,:,i]
        # eigval = eigvals(inv(Q_))
        # radius = sqrt.(1 ./ eigval)
        # println("radius of x,y,theta: ", radius)
        A = [1 0 0; 0 1 0]
        Q_proj = A * Q_ * A'
        Q_inv = inv(Q_proj)
        eigval, eigvec = eigen(Q_inv)
        radius = sqrt.(1 ./ eigval)
        # println("radius of x and y: ", radius)
        rnew = eigvec * [radius[1]; 0]
        angle = atan(rnew[2], rnew[1])
        push!(radius_list, radius)
        push!(angle_list, angle)
    end
    return radius_list, angle_list
end

function propagate_multiple_FOH(model::FunnelDynamics,dynamics::Dynamics,
    x::Matrix,u::Matrix,T::Vector,
    Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3},
    flag_single::Bool=false)
    N = size(x,2) - 1
    ix = model.ix
    iu = model.iu
    iq = model.iq
    iy = model.iy

    idx_x = 1:ix
    idx_q = (ix+1):(ix+iq)

    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        ym = p[3]
        yp = p[4]
        zm = p[5]
        zp = p[6]
        dt = p[7]

        alpha = (dt - t) / dt
        beta = t / dt

        u_ = alpha * um + beta * up
        y_ = alpha * ym + beta * yp
        z_ = alpha * zm + beta * zp

        x_ = V[idx_x]
        q_ = V[idx_q]

        # traj terms
        f = forward(dynamics,x_,u_)
        A,B = diff(dynamics,x_,u_)
        # funl terms
        F = forward(model,q_,y_,z_,A,B)

        dxdt = f
        dqdt = F
        dV = [dxdt;dqdt]
        out .= dV[:]
    end

    tprop = []
    xprop = []
    uprop = []
    qprop = []
    yprop = []
    zprop = []
    Qfwd = zeros(size(Q))
    Qfwd[:,:,1] .= Q[:,:,1]
    for i = 1:N
        if flag_single == true
            V0 = [x[:,i];vec(Qfwd[:,:,i])][:]
        else
            V0 = [x[:,i];vec(Q[:,:,i])][:]
        end

        um = u[:,i]
        up = u[:,i+1]
        ym = vec(Y[:,:,i])
        yp = vec(Y[:,:,i+1])
        zm = vec(Z[1:ix,:,i])
        if typeof(model) == LinearFOH
            zp = vec(Z[:,:,i])
        elseif typeof(model) == LinearSOH
            zp = vec(Z[ix+1:2*ix,:,i])
        else
            zp = vec(Z[:,:,i+1])
        end
        dt = T[i]

        prob = ODEProblem(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,dt))
        sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12;verbose=false);

        tode = sol.t[1:end-1]
        uode = zeros(iu,size(tode,1))
        yode = zeros(iy,size(tode,1))
        zode = zeros(iq,size(tode,1))
        for idx in 1:length(tode)
            alpha = (dt - tode[idx]) / dt
            beta = tode[idx] / dt
            uode[:,idx] .= alpha * um + beta * up
            yode[:,idx] .= alpha * ym + beta * yp
            zode[:,idx] .= alpha * zm + beta * zp
        end
        ode = stack(sol.u)
        xode = ode[idx_x,1:end-1]
        qode = ode[idx_q,1:end-1]
        if i == 1
            tprop = tode
            xprop = xode
            uprop = uode
            qprop = qode
            yprop = yode
            zprop = zode
        else 
            tprop = vcat(tprop,sum(T[1:i-1]).+tode)
            xprop = hcat(xprop,xode)
            uprop = hcat(uprop,uode)
            qprop = hcat(qprop,qode)
            yprop = hcat(yprop,yode)
            zprop = hcat(zprop,zode)
        end
        Qfwd[:,:,i+1] = reshape(ode[idx_q,end],(ix,ix))
    end
    tprop = vcat(tprop,[sum(T[1:N])])
    xprop = hcat(xprop,x[:,end])
    uprop = hcat(uprop,u[:,end])
    qprop = hcat(qprop,vec(Q[:,:,end]))
    yprop = hcat(yprop,vec(Y[:,:,end]))
    zprop = hcat(zprop,vec(Z[:,:,end]))

    Qprop = zeros(ix,ix,length(tprop))
    Yprop = zeros(iu,Int64(iy/iu),length(tprop))
    Zprop = zeros(ix,ix,length(tprop))
    for i in 1:length(tprop)
        Qprop[:,:,i] .= reshape(qprop[:,i],(ix,ix))
        Yprop[:,:,i] .= reshape(yprop[:,i],(iu,Int64(iy/iu)))
        Zprop[:,:,i] .= reshape(zprop[:,i],(ix,ix))
    end
    return Qfwd,tprop,xprop,uprop,Qprop,Yprop,Zprop
end

function propagate_from_funnel_entry(x0::Vector,model::FunnelDynamics,dynamics::Dynamics,
    xnom::Matrix,unom::Matrix,Tnom::Vector,
    Q::Array{Float64,3},Y::Array{Float64,3},Z::Array{Float64,3})
    N = size(xnom,2) - 1
    ix = model.ix
    iu = model.iu
    iq = model.iq
    iy = model.iy

    idx_x = 1:ix
    idx_xnom = ix+1:2*ix
    idx_q = (2*ix+1):(2*ix+iq)

    function dvdt(out,V,p,t)
        um = p[1]
        up = p[2]
        ym = p[3]
        yp = p[4]
        zm = p[5]
        zp = p[6]
        dt = p[7]
        km = p[8]
        kp = p[9]

        alpha = (dt - t) / dt
        beta = t / dt

        unom_ = alpha * um + beta * up
        y_ = alpha * ym + beta * yp
        z_ = alpha * zm + beta * zp
        k_ = alpha * km + beta * kp

        x_ = V[idx_x]
        xnom_ = V[idx_xnom]
        q_ = V[idx_q]

        Q_ = reshape(q_,(ix,ix))
        Y_ = reshape(y_,(iu,ix))
        K_ = Y_ * inv(Q_)
        # K_ = reshape(k_,(iu,ix))

        u_ = unom_ + K_ * (x_ - xnom_)

        # traj terms
        f = forward(dynamics,x_,u_)
        fnom = forward(dynamics,xnom_,unom_)
        A,B = diff(dynamics,x_,u_)
        # funl terms
        F = forward(model,q_,y_,z_,A,B)

        dV = [f;fnom;F]
        out .= dV[:]
    end

    xfwd = zeros(size(xnom))
    xfwd[:,1] .= x0
    tprop = []
    xprop = []
    uprop = []
    for i = 1:N
        V0 = [xfwd[:,i];xnom[:,i];vec(Q[:,:,i])][:]
        um = unom[:,i]
        up = unom[:,i+1]
        ym = vec(Y[:,:,i])
        yp = vec(Y[:,:,i+1])
        km = vec(Y[:,:,i] * inv(Q[:,:,i]))
        kp = vec(Y[:,:,i+1] * inv(Q[:,:,i+1]))
        zm = vec(Z[1:ix,:,i])
        if typeof(model) == LinearFOH
            zp = vec(Z[:,:,i])
        elseif typeof(model) == LinearSOH
            zp = vec(Z[ix+1:2*ix,:,i])
        else
            zp = vec(Z[:,:,i+1])
        end
        dt = Tnom[i]

        prob = ODEProblem(dvdt,V0,(0,dt),(um,up,ym,yp,zm,zp,dt,km,kp))
        sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9;verbose=false);

        tode = sol.t
        ode = stack(sol.u)
        xode = ode[idx_x,:]
        xnomode = ode[idx_xnom,:]
        qode = ode[idx_q,:]
        uode = zeros(iu,size(tode,1))
        for idx in 1:length(tode)
            alpha = (dt - tode[idx]) / dt
            beta = tode[idx] / dt

            unom_ = alpha * um + beta * up
            y_ = alpha * ym + beta * yp
            x_ = xode[:,idx]
            xnom_ = xnomode[:,idx]
            q_ = qode[:,idx]

            Q_ = reshape(q_,(ix,ix))
            Y_ = reshape(y_,(iu,ix))
            K_ = Y_ * inv(Q_)
            uode[:,idx] .= unom_ + K_ * (x_ - xnom_)
        end
        if i == 1
            tprop = tode
            xprop = xode
            uprop = uode
        else 
            tprop = vcat(tprop,sum(Tnom[1:i-1]).+tode)
            xprop = hcat(xprop,xode)
            uprop = hcat(uprop,uode)
        end
        xfwd[:,i+1] = xode[:,end]
    end
    return xfwd,tprop,xprop,uprop
end

using JuMP
function Lipschitz_estimation_around_traj(N::Int,num_sample::Int,
    xnom::Matrix{Float64},unom::Matrix{Float64},
    dynamics::Dynamics,Qnode::Array{Float64,3},Rnode::Array{Float64,3})
    gamma_sample = zeros(num_sample,N+1)

    for idx in 1:N+1
        for j in 1:num_sample
            sqrt_Q = sqrt(Qnode[:,:,idx])
            sqrt_R = sqrt(Rnode[:,:,idx])

            z = randn(ix)
            z .= z / norm(z)
            eta_sample = sqrt_Q * z

            z = randn(iu)
            z .= z / norm(z)
            xii_sample = sqrt_R * z

            z = randn(iw)
            z .= z / norm(z)
            w_sample = z

            # K = Knode[:,:,idx]
            x_ = xnom[:,idx] + eta_sample
            u_ = unom[:,idx] + xii_sample
            w_ = w_sample

            A,B,F = diff_ABF(dynamics,xnom[:,idx],unom[:,idx])

            eta_dot = forward_uncertain(dynamics,x_,u_,w_) - forward(dynamics,xnom[:,idx],unom[:,idx])
            LHS = eta_dot - A * eta_sample - B * xii_sample - F * w_sample
            delta_q = dynamics.Co * eta_sample +  dynamics.Do * xii_sample + dynamics.Go * w_sample

            model = Model(Mosek.Optimizer)
            # set_optimizer_attribute(model, "verbose", false) # Turn off verbosity for Clarabel
            set_optimizer_attribute(model, "MSK_IPAR_LOG", 0) # Turn off verbosity for Mosek
            @variable(model, Delta[1:dynamics.iphi,1:dynamics.iq])
            @constraint(model,LHS == dynamics.Eo * Delta * delta_q)
            @objective(model,Min,dot(vec(Delta),vec(Delta)))
            optimize!(model)

            gamma_sample[j,idx] = opnorm(value.(Delta),2)
        end
    end
    return maximum(gamma_sample,dims=1)[:]
end