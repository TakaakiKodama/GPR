module GPLM
using Random
using LinearAlgebra
using Plots

function VisualizeGPLM(GPR)
    plot(GPR.x_test,GPR.predictMean,label="Predicted curve")
#     if x!=0 && y!=[]
#         plot!(GPR.x_test,y.(GPR.x_test),label="Theorcetical curve", dpi=300)
#     end
    plot!(GPR.X_train,GPR.Y_train,seriestype=:scatter, label="toydata")
    plot!(GPR.x_test,GPR.predictMean+2 .*GPR.predictVariance, label="+2signma")
    plot!(GPR.x_test,GPR.predictMean-2 .*GPR.predictVariance,label="-2signma")
end


struct ToyData
    X
    Y
end
function ToyData(f,dim_i,dim_o;min=0,max=1,N::Int64=100,messiness=[1 for i in 1:dim_o])
	X=[]
	Y=[]
	for i in 1:1:N
		x=[min+rand(1)[1]*(max-min) for i in 1:dim_i]'
		if dim_o==1
			y=f(x)+randn(1)[1]*messiness(x)[1]
		else
			y=f(x)+[randn(1)[1]*messiness(x)[i] for i in 1:dim_o]'
		end
		push!(Y,y)
		push!(X,x)
	end
	X=vcat(X...)
	Y=vcat(Y...)
	return ToyData(X,Y)
end

# CH1

function MultipleRegression(X,y)
	X=hcat([1 for i in 1:size(X,1)],X)
	w=inv(X'*X)*X'*y
	return(w)
end

function PolynomialRegression(X,y,f)
	if typeof(f)==Int64
		ϕ(x)=[x[1]^n for n in 0:order]
	else
		ϕ=f
	end
	Φ=[]
	for i in 1:size(X,1)
		push!(Φ,ϕ(X[i,:])')
	end
	Φ=vcat(Φ...)
	w=inv(Φ'*Φ)*Φ'*y
	return(w,ϕ)
end

function RidgeRegression(X,y;a=0.1)
	X=hcat([1 for i in 1:size(X,1)],X)
	w=inv(X'*X+a*I)*X'*y
	return(w)
end

# CH2
function ExpectedValue(x;p=1/length(x))
	return(sum([el*p for el in x]))
end

function MeanVector(x)
	return([ExpectedValue(x) for i in 1:length(x)])
end

function MeanMatrix(X)
	return(hcat([GPLM.MeanVector(X[:,i]) for i in 1:size(X,2)]...))
end

function CovarianceMatrix(X)
	Σ=zeros(size(X,1),size(X,1))
	M=MeanMatrix(X)
	for i in 1:size(X,2)
		for j in 1:i
			Σ[i,j]=Σ[j,i]
		end
		for j in i:size(X,2)
			Σ[i,j]=GPLM.ExpectedValue([(X[k,i]-M[k,i])*(X[k,j]-M[k,j]) for k in 1:size(X,1)])
		end
	end
	return(Σ)
end

function GaussianDistribution(x;μ=[0 for i in 1:length(x)],Σ=I)
	return(exp(-1/2*(x-μ)'*inv(Σ)*(x-μ))/((2π)^length(x)*det(Σ))^(1/2))
end

# CH3

function GaussianKernel(x,x_,;th1=1,th2=1)
	return(th1*exp(-(x-x_)*(x-x_)'/th2)[1])
end

function ExponentialKernel(x,x_,;th=1)
	return(exp(-√((x-x_)*(x-x_)')/th)[1])
end

function PeriodicKernel(x,x_,;th1=1,th2=1)
	return(exp(th1*cos(√((x-x_)*(x-x_)')/th2))[1])
end

function KernelMatrix(X,k;observationn_noise=0)
	K=zeros(size(X,1),size(X,1))
	for i in 1:size(X,1)
		for j in 1:i
			K[i,j]=K[j,i]
		end
		for j in i:size(X,1)
			if i==j
				N=observationn_noise
			else
				N=0
			end
			K[i,j]=k(X[i,:],X[j,:])+N
		end
	end
	return(K)
end

struct GaussianProcessRegression
    X_train
    Y_train
    x_test
    predictMean
    predictVariance
    learnedKernelMatrix
    learnedInvKernelMatrix
end

function GaussianProcessRegression(X_train,Y_train,k,x_test,observationn_noise)
	K=KernelMatrix(X_train,k,observationn_noise=observationn_noise)
	K_=inv(K)
	yy=K_*Y_train
	μ=[]
	σ=[]
    for i in x_test
        k_=[k(j,i) for j in X_train]
        s=k(i,i)
        mu=(k_'*yy)[1]
        var=s-(k_'*K_*k_)[1]
        push!(μ,mu)
        push!(σ,var)
    end
    predictMean=μ
    predictVariance=σ
    learnedKernelMatrix=K
    learnedInvKernelMatrix=K_
    return GaussianProcessRegression(X_train,Y_train,x_test,predictMean,predictVariance,learnedKernelMatrix,learnedInvKernelMatrix)

end

end
