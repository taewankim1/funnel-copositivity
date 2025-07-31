using Interpolations
using LinearAlgebra
using Plots
using DifferentialEquations
using StaticArrays
function print_jl(x,flag_val = false)
    println("Type is $(typeof(x))")
    println("Shape is $(size(x))")
    if flag_val == true
        println("Value is $(x)")
    end
end


function plot_ellipsoid_3D(
                        plot::Union{Nothing,Plots.Plot},
                        center::AbstractVector{<:Real},
                        Q::AbstractMatrix{<:Real};
                        n::Int=60, color=:deepskyblue, alpha=0.6, label=nothing, solid=true)

    @assert size(Q) == (3,3)            "Q must be 3×3"
    @assert length(center) == 3         "center must have 3 elements"

    # linear map that turns unit sphere into the ellipsoid
    λ, V = eigen(Symmetric(Q))
    T    = V * Diagonal(sqrt.(λ))

    # polar grid on the unit sphere
    ϕ = range(0, 2π; length=n)
    θ = range(0, π ; length=n)
    xs = zeros(n, n); ys = similar(xs); zs = similar(xs)

    @inbounds for i in eachindex(ϕ), j in eachindex(θ)
        s = @SVector [cos(ϕ[i])*sin(θ[j]),
                      sin(ϕ[i])*sin(θ[j]),
                      cos(θ[j])]
        p = T*s .+ center
        xs[i,j], ys[i,j], zs[i,j] = p
    end

    if label !== nothing
        showlegend = true
    end

    if solid
        # ───── filled surface ──────────────────────────────────────────
        surface!(plot, xs, ys, zs;
                 linewidth  = 0,          # no mesh lines
                 color      = color,
                 alpha      = alpha,
                 label      = label,
                 aspect_ratio = :equal,
                 showlegend = false,
                 colorbar   = false)
    else
        # ───── wire-frame shell only ───────────────────────────────────
        plot!(plot, xs, ys, zs;
              st         = :wireframe,    # series-type
              linecolor  = color,
              linewidth  = 1.4,
              label      = label,
              aspect_ratio = :equal,
              colorbar   = false)
    end
end

function plot_ellipse(plot,Q::Matrix,xbar::Vector,color;alpha=0.3,linewidth=2,fill=true,label=nothing,idx1=1,idx2=2)
    θ = range(0, 2pi + 0.05; step = 0.05)
    x_y = √Q[[idx1,idx2],[idx1,idx2]] * hcat(cos.(θ), sin.(θ))' .+ xbar[[idx1,idx2]]
    plot!(plot, x_y[1, :], x_y[2, :], c = color,linewidth=linewidth,label=label)
    plot!(plot, x_y[1, :], x_y[2, :], label = nothing,fill=fill, fillcolor=color,alpha=alpha)
    plot!(legendfontsize=12)
end

include("dynamics.jl")
function matrix_to_vector(matrix::Array)
    return [vec(col) for col in eachcol(matrix)]
end

function propagate_multiple_FOH(model::Dynamics,x::Matrix,u::Matrix,T::Vector)
    N = size(x,2) - 1
    ix = size(x,1)
    iu = size(u,1)

    function model_wrapper!(f,x,p,t)
        um = p[1]
        up = p[2]
        dt = p[3]
        alpha = 1 - t
        beta = t
        u1 = alpha*um + beta*up
        f .= dt*forward(model,x,u1)
    end

    tspan = (0,1)
    tprop = []
    xprop = []
    xnew = zeros(size(x))
    xnew[:,1] .= x[:,1]
    for i in 1:N
        prob = ODEProblem(model_wrapper!,x[:,i],tspan,(u[:,i],u[:,i+1],T[i]))
        sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9;verbose=false);
        tode = sol.t
        xode = stack(sol.u)
        if i == 1
            tprop = T[i]*tode
            xprop = xode
        else 
            tprop = vcat(tprop,sum(T[1:i-1]).+T[i]*tode)
            xprop = hcat(xprop,xode)
        end
        xnew[:,i+1] .= xode[:,end]
    end
    return xnew,tprop,xprop
end