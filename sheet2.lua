require 'torch'
require 'optim'
require 'nn'

-- MOVIES: Legally Blond; Matrix; Bourne Identity; You've Got Mail;
--		   The Devil Wears Prada; The Dark Knight; The Lord of the Rings

P = torch.Tensor{{0,0,-1,0,-1,1,1}, {-1,1,1,-1,0,1,1}, {0,1,1,0,0,-1,1},
				 {-1,1,1,0,0,1,1}, {0,1,1,0,0,1,1}, {1,-1,1,1,1,-1,0},
				 {-1,1,-1,0,-1,0,1}, {0,-1,0,1,1,-1,-1}, {0,0,-1,1,1,0,-1}}

reg = 0.1; f=2; m = 9; n=7; -- parameters; reg for regularization, f factors

--random init; X = (m x f), Y = (f x n)
--given code initializes with uniform random from [0,1)
X = torch.mul(torch.add(torch.ones(m,f),-(torch.mul((torch.rand(m,f)),2))),0.1)
Y = torch.mul(torch.add(torch.ones(f,n),-(torch.mul((torch.rand(f,n)),2))),0.1)
C = torch.abs(P)  --weight matrix
epochs = 100

--alternating weighted ridge regression:
for epoch = 1,epochs do
	-- solve for X keeping Y fixed; user u has set of weights Cu:
	for u = 1,m do
	Cu = torch.diag(C[u])					--construct matrix Cu
	K = (Y*Cu)*(Y:t()) + (torch.mul(torch.eye(f),reg))
	X[u]=(torch.inverse(K)*Y*Cu*P[u])
	end
	
	-- solve for Y keeping X fixed; user u has set of weights Ci:
	--Pt = P:t()
	Yt = Y:t()								--transpose to index over rows
	for i = 1,n do
	Ci = torch.diag((C:t())[i])				--construct matrix Ci
	K = ((X:t())*Ci*X + (torch.mul(torch.eye(f),reg)))
	Pi = P:select(2,i)						--get ith column of P
	Yt[i]=(torch.inverse(K)*(X:t())*Ci*Pi)
	end
	Y = Yt:t()								--transpose back
end
print(X*Y)	-- result
