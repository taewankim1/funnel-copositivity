# Define the Scaling struct with an inner constructor
mutable struct Scaling
    Sx::Matrix{Float64}
    iSx::Matrix{Float64}
    sx::Vector{Float64}
    Su::Matrix{Float64}
    iSu::Matrix{Float64}
    su::Vector{Float64}
    Slam::Matrix{Float64}
    S_sigma::Float64
    
    # Define the inner constructor
    function Scaling(xmin, xmax, umin, umax, slam1, slam2)
        Sx, iSx, sx, Su, iSu, su = compute_scaling(xmin, xmax, umin, umax)
        Slam = diagm([slam1,slam2])
        S_sigma = 1.0
        new(Sx, iSx, sx, Su, iSu, su, Slam,S_sigma)
    end
end

# Define the compute_scaling function
function compute_scaling(xmin, xmax, umin, umax)
    tol_zero = 1e-10

    x_intrvl = [0, 1]
    u_intrvl = [0, 1]
    x_width = x_intrvl[2] - x_intrvl[1]
    u_width = u_intrvl[2] - u_intrvl[1]

    Sx = (xmax - xmin) / x_width
    Sx[Sx .< tol_zero] .= 1
    Sx = diagm(Sx)
    iSx = inv(Sx)
    sx = xmin - x_intrvl[1] * diag(Sx)
    @assert size(sx, 2) == 1

    Su = (umax - umin) / u_width
    Su[Su .< tol_zero] .= 1
    Su = diagm(Su)
    iSu = inv(Su)
    su = umin - u_intrvl[1] * diag(Su)
    @assert size(su, 2) == 1

    return Sx, iSx, sx, Su, iSu, su
end
