### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 6f3dcbf4-1b5e-4f64-995f-db9b08c381e2
# function _getHamiltonian(B::Float64, Ω::ComplexF64, δ::Float64, rabiRatio::Float64)
# 	Γ::Float64 = 2π*5.746e6  # Decay Rate of Excited State
#     μ::Float64 = 2π*0.7e6
#     H = zeros(3, 3) * 0im
#     H[1, 1] = μ * B
#     H[2, 2] = -μ * B + δ
#     H[1, 3] = Ω / 2 * rabiRatio
#     H[2, 3] = Ω / 2
#     H[3, 1] = Ω' / 2 * rabiRatio
#     H[3, 2] = Ω' / 2
#     H[3, 3] = -1im/2 * Γ
#     return H
# end

# ╔═╡ 5a13e050-f870-420d-95c0-16387bdbc700
# function _∂tρ(ρ::Matrix{ComplexF64}, B::Float64, ReΩ::Float64, ImΩ::Float64, δ::Float64)
# 	Γ::Float64 = 2π*5.746e6  # Decay Rate of Excited State
# 	Ω = ReΩ + 1im*ImΩ
# 	rabiRatio::Float64 = 1
#     H = _getHamiltonian(B, Ω, δ, rabiRatio)
#     ∂tρ = -1im * (H*ρ - ρ*H')
#     ∂tρ[1, 1] = ∂tρ[1, 1] + ρ[3, 3] * Γ * rabiRatio^2 / (rabiRatio^2 + 1)
#     ∂tρ[2, 2] = ∂tρ[2, 2] + ρ[3, 3] * Γ / (rabiRatio^2 + 1)
#     ∂tρ
# end

# ╔═╡ 0963af20-c448-11ec-2a04-a3ea18c72664
"""
ρ2vec(ρ::Matrix{ComplexF64})::Vector{Float64}

Vectorize the density matrix `\rho`, please refer [vec2ρ](@Ref)

# Arguments
- `ρ::Matrix{ComplexF64}`: Density matrix

# Examples:

```julia-repl
julia> begin
	ρ = zeros(3, 3) * 1im
	ρ[1, 1] = 1/2
	ρ[2, 2] = 1/2
	ρ[1, 2] = 1 + 2im
	ρ[2, 1] = 1 - 2im
	ρ[3, 3] = 0
end
julia> vec = ρ2vec(ρ)
```
"""
function ρ2vec(ρ::Matrix{ComplexF64})::Vector{Float64}
	sz = size(ρ)[1]
	vec = zeros(sz*sz)
	for i = 1:sz
		vec[i] = real(ρ[i, i])
	end
	index = sz + 1
	for i = 1:(sz-1)
		for j = (i+1) : sz
			vec[index] = real(ρ[i, j])
			vec[index+1] = imag(ρ[i, j])
			index += 2
		end
	end
	vec
end

# ╔═╡ 5ef0673f-1a85-410c-9d7f-2c00e4cc8ad9
"""
vec2ρ(vec::Vector{Float64})::Matrix{ComplexF64}

Reconstruct density matrix from a vector, please refer [ρ2vec](@Ref)

# Arguments

- `vec::Vector{Float64}`: Vectorized of density matrix

# Examples
```julia-repl
julia> begin
	ρ = zeros(3, 3) * 1im
	ρ[1, 1] = 1/2
	ρ[2, 2] = 1/2
	ρ[1, 2] = 1 + 2im
	ρ[2, 1] = 1 - 2im
	ρ[3, 3] = 0
end
julia> vec = ρ2vec(ρ)
julia> ρ2 = vec2ρ(vec)
julia> ρ2 - ρ
```
"""
function vec2ρ(vec::Vector{Float64})::Matrix{ComplexF64}
	sz = floor(Int64, sqrt(size(vec)[1]))
	ρ = zeros(sz, sz) * 1im
	for i = 1:sz
		ρ[i, i] = vec[i]
	end
	index = sz + 1
	for i = 1:(sz - 1)
		for j = (i+1) : sz
			ρ[i, j] = vec[index] + 1im*vec[index + 1]
			ρ[j, i] = vec[index] - 1im*vec[index + 1]
			index += 2
		end
	end
	ρ
end

# ╔═╡ 1272856e-e6b6-4318-b1a7-6972049f0f09
"""
@getcoefficient(coe, sz, masterFun, args...)

get the coefficient of an Master equation

# Arguments:

- `coe`: The coefficient dictionary.
- `sz`: The size of the system, number of energy levels
- `masterFun`: The master equation

# Examples:

```julia-repl
julia> function _getHamiltonian(B::Float64, Ω::ComplexF64, δ::Float64, rabiRatio::Float64)
	Γ::Float64 = 2π*5.746e6  # Decay Rate of Excited State
    μ::Float64 = 2π*0.7e6
    H = zeros(3, 3) * 0im
    H[1, 1] = μ * B
    H[2, 2] = -μ * B + δ
    H[1, 3] = Ω / 2 * rabiRatio
    H[2, 3] = Ω / 2
    H[3, 1] = Ω' / 2 * rabiRatio
    H[3, 2] = Ω' / 2
    H[3, 3] = -1im/2 * Γ
    return H
end
julia> function _∂tρ(ρ::Matrix{ComplexF64}, B::Float64, ReΩ::Float64, ImΩ::Float64, δ::Float64)
	Γ::Float64 = 2π*5.746e6  # Decay Rate of Excited State
	Ω = ReΩ + 1im*ImΩ
	rabiRatio::Float64 = 1
    H = _getHamiltonian(B, Ω, δ, rabiRatio)
    ∂tρ = -1im * (H*ρ - ρ*H')
    ∂tρ[1, 1] = ∂tρ[1, 1] + ρ[3, 3] * Γ * rabiRatio^2 / (rabiRatio^2 + 1)
    ∂tρ[2, 2] = ∂tρ[2, 2] + ρ[3, 3] * Γ / (rabiRatio^2 + 1)
    ∂tρ
end
julia> @getcoefficient coe 3 _∂tρ B ReΩ ImΩ δ 
```
"""
macro getcoefficient(coe, sz, masterFun, args...)
	sysSize = eval(sz) * eval(sz)
	argSize = length(args)
	# coeDict = Dict(arg => zeros(sysSize, sysSize) for arg in args)
	# for arg in args
	# 	str1 = "$masterFun("
	# 	println(str1)
	# 	for i = 1:sysSize
	# 		vec = zeros(sysSize)
	# 		vec[i] = 1
	# 		# coeDict[arg]
	# 	end
	# end
	exprs = []
	expr1 = :($(esc(coe)) = Dict(arg => zeros($(esc(sysSize)), $(esc(sysSize))) for arg in [:M0, $(esc(args))...] ))
	push!(exprs, expr1)

	expr2 = quote
		argArr = zeros($(esc(argSize)))
		argName = :M0
		for i=1:$(esc(sysSize))
			vec = zeros($(esc(sysSize)))
			vec[i] = 1
			ρ = vec2ρ(vec)
			dρ = $(esc(masterFun))(ρ, argArr...)
			$(esc(coe))[argName][:,i] = ρ2vec(dρ)
		end
	end
	push!(exprs, expr2)
	
	
	for a = 1:argSize
		expri = quote
			argArr = zeros($(esc(argSize)))
			argArr[$a] = 1
			argName = $(esc(args))[$a]
			for i=1:$(esc(sysSize))
				vec = zeros($(esc(sysSize)))
				vec[i] = 1
				ρ = vec2ρ(vec)
				dρ = $(esc(masterFun))(ρ, argArr...)
				$(esc(coe))[argName][:,i] = ρ2vec(dρ)
			end
			$(esc(coe))[argName] = $(esc(coe))[argName] - $(esc(coe))[:M0]
		end
	push!(exprs, expri)
	end

	push!(exprs, :(nothing))
	
	quote
		$(exprs...)
	end
end

# ╔═╡ aa99fb3b-2ca3-4eab-8e6b-88b9d75399dd
export @getcoefficient

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[deps]
"""

# ╔═╡ Cell order:
# ╠═6f3dcbf4-1b5e-4f64-995f-db9b08c381e2
# ╠═5a13e050-f870-420d-95c0-16387bdbc700
# ╠═0963af20-c448-11ec-2a04-a3ea18c72664
# ╠═5ef0673f-1a85-410c-9d7f-2c00e4cc8ad9
# ╠═1272856e-e6b6-4318-b1a7-6972049f0f09
# ╠═aa99fb3b-2ca3-4eab-8e6b-88b9d75399dd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
