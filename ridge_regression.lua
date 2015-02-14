require 'torch'
require 'optim'
require 'nn'
require 'gnuplot'         -- plot outputs as well

model   = nn.Sequential() -- define container as sequential; similar to previous problem
ninputs = 2; noutputs = 1 -- try same problem

-- data generation

-- quadratic polynomial for data generation: f(x,y) = a + bx + cx^2; plottable
a = 10; b = 5; c = 3 -- set generation parameters here
nTrain = 20	     -- set number of training instances
nTest  = 5
data = torch.Tensor(nTrain,1); labels = torch.Tensor(nTrain,1); 
testData = torch.Tensor(nTest,1); testLabels = torch.Tensor(nTest,1);

for i=1,nTrain do
	local x = torch.uniform(2,6)
	local noise = torch.normal(0,1) --add Gaussian noise; mean zero, variance 1 
	data[i][1] = x
	labels[i][1]=a + b*x+c*x*x+noise
end

for i=1,nTest do
	local x = torch.uniform(2,6)
	local noise = torch.normal(0,1) --add Gaussian noise; mean zero, variance 1 
	testData[i][1] = x
	testLabels[i][1]=a + b*x+c*x*x+noise
end

model:add(nn.Linear(ninputs, noutputs)) -- define the only module
-- we use a linear classifier, with a basis function phi

local function quadphi(x)			-- quadratic basis function
	return torch.Tensor{1,x,x*x}
end
local function cubephi(x)			-- cubic basis function
	return torch.Tensor{1,x,x*x,x*x*x}
end
local function quartphi(x)			-- quartic basis function
	return torch.Tensor{1,x,x*x,x*x*x,x*x*x*x}
end

-- kernelize data
local quadPhi = torch.Tensor(nTrain,3)
for i=1,nTrain do
	local phirow = quadphi(data[i][1])
	for j=1,3 do
		quadPhi[i][j]=phirow[j]
	end
end
local cubePhi = torch.Tensor(nTrain,4)
for i=1,nTrain do
	local phirow = cubephi(data[i][1])
	for j=1,4 do
		cubePhi[i][j]=phirow[j]
	end
end
local quartPhi = torch.Tensor(nTrain,5)
for i=1,nTrain do
	local phirow = quartphi(data[i][1])
	for j=1,5 do
		quartPhi[i][j]=phirow[j]
	end
end

local regularizer = torch.mul(torch.eye(nTrain),0.001)
local thetaQuad   = torch.inverse((quadPhi:t()*quadPhi)+regularizer)*(quadPhi:t())*labels
local thetaCube   = torch.inverse((cubePhi:t()*cubePhi)+regularizer)*(cubePhi:t())*labels
local thetaQuart  = torch.inverse((quartPhi:t()*quartPhi)+regularizer)*(quartPhi:t())*labels

y = data:narrow(2,1,1) --slice first column		   --for linear model
X = torch.cat(torch.ones((#data)[1],1),data:narrow(2,2,2)) --slice rest of it and add bias
XT=X:transpose(1,2)    --transpose X by swapping width (1) & height (2)
print('LSParams')
LSParams = (torch.inverse(XT * X))*XT*y
print(LSParams)

local quadPhiTest = torch.Tensor(nTrain,3)
for i=1,nTest do
	local phirow = quadphi(testData[i][1])
	for j=1,3 do
		quadPhiTest[i][j]=phirow[j]
	end
end
local cubePhiTest = torch.Tensor(nTrain,4)
for i=1,nTest do
	local phirow = cubephi(testData[i][1])
	for j=1,4 do
		cubePhiTest[i][j]=phirow[j]
	end
end
local quartPhiTest = torch.Tensor(nTrain,5)
for i=1,nTest do
	local phirow = quartphi(testData[i][1])
	for j=1,5 do
		quartPhiTest[i][j]=phirow[j]
	end
end

biasedDataTest = torch.cat(torch.ones((#testData)[1],1), testData)
local quadPred = quadPhiTest:t() * thetaQuad    --quadratic model
local cubePred = cubePhiTest:t() * thetaCube    --cubic model
local quartPred= quartPhiTest:t()* thetaQuart   --quartic model
local LSPred  = biasedDataTest * LSParams --linear model
print(LSPredictions)

