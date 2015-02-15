require 'torch'
require 'optim'
require 'nn'
require 'gnuplot'         -- plot outputs as well

-- data generation
-- quadratic polynomial for data generation: f(x,y) = a + bx + cx^2; plottable
a = 2; b = -3; c = 1 -- set generation parameters here
nTrain = 1000	     -- set number of training instances
nTest  = 50
data = torch.Tensor(nTrain,1); labels = torch.Tensor(nTrain,1); 
testData = torch.Tensor(nTest,1); testLabels = torch.Tensor(nTest,1);

for i=1,nTrain do
	local x = torch.uniform(1,5)
	local noise = torch.normal(0,0.1) --add Gaussian noise; mean zero, variance 0.1
	data[i][1] = x
	labels[i][1]=a + b*x+c*x*x+noise
end

for i=1,nTest do
	local x = torch.uniform(1,5)
	local noise = torch.normal(0,0.1) --add Gaussian noise; mean zero, variance 0.1
	testData[i][1] = x
	testLabels[i][1]=a + b*x+c*x*x+noise
end


local function quadphi(x)			-- quadratic basis function
	return torch.Tensor{1,x,x*x}
end
local function cubephi(x)			-- cubic basis function
	return torch.Tensor{1,x,x*x,x*x*x}
end
local function quartphi(x)			-- quartic basis function
	return torch.Tensor{1,x,x*x,x*x*x,x*x*x*x}
end

local quadPhi = torch.Tensor(nTrain,3)	--extract features using
local cubePhi = torch.Tensor(nTrain,4)  --the phi functions
local quartPhi = torch.Tensor(nTrain,5)

for i=1,nTrain do
	local phirow = quadphi(data[i][1])
	for j=1,3 do
		quadPhi[i][j]=phirow[j]
	end

	local phirow = cubephi(data[i][1])
	for j=1,4 do
		cubePhi[i][j]=phirow[j]
	end

	local phirow = quartphi(data[i][1])
	for j=1,5 do
		quartPhi[i][j]=phirow[j]
	end
end

local regularizerQuad = torch.mul(torch.eye(3),0.0001) --regularize
local regularizerCube = torch.mul(torch.eye(4),0.0001)
local regularizerQuar = torch.mul(torch.eye(5),0.0001)

local thetaQuad  = torch.inverse(((quadPhi:t())*quadPhi)+regularizerQuad)*(quadPhi:t())*labels
local thetaCube  = torch.inverse(((cubePhi:t())*cubePhi)+regularizerCube)*(cubePhi:t())*labels
local thetaQuart = torch.inverse(((quartPhi:t())*quartPhi)+regularizerQuar)*(quartPhi:t())*labels

X = torch.cat(torch.ones((#data)[1],1),data) --and add bias for linear model
XT=X:transpose(1,2)    --transpose X by swapping width (1) & height (2)
LSParams = (torch.inverse(XT * X))*XT*labels --linear parameters

local quadPhiTest = torch.Tensor(nTest,3)
local cubePhiTest = torch.Tensor(nTest,4)
local quartPhiTest = torch.Tensor(nTest,5)

for i=1,nTest do
	local phirow = quadphi(testData[i][1])
	for j=1,3 do
		quadPhiTest[i][j]=phirow[j]
	end

	local phirow = cubephi(testData[i][1])
	for j=1,4 do
		cubePhiTest[i][j]=phirow[j]
	end

	local phirow = quartphi(testData[i][1])
	for j=1,5 do
		quartPhiTest[i][j]=phirow[j]
	end
end

biasedDataTest = torch.cat(torch.ones((#testData)[1],1), testData)

local quadPred = quadPhiTest * thetaQuad   --quadratic model
local cubePred = cubePhiTest * thetaCube    --cubic model
local quartPred= quartPhiTest* thetaQuart   --quartic model
local LSPred  = biasedDataTest * LSParams 		--linear model

print("Linear model predictions")
print(LSPred)
print("Quadratic model predictions")
print(quadPred)
print("Cubic model predictions")
print(cubePred)
print("Quartic model predictions")
print(quartPred)
print("Actual values")
print(testLabels)

print("Linear parameters")
print(LSParams)
print("Quadratic parameters")
print(thetaQuad)
print("Cubic parameters")
print(thetaCube)
print("Quartic parameters")
print(thetaQuart)

gnuplot.plot({torch.cat(testData,quadPred),'+'}, {torch.cat(testData,testLabels), '+'})
gnuplot.plot({torch.cat(testData,cubePred),'+'}, {torch.cat(testData,testLabels), '+'})
gnuplot.plot({torch.cat(testData,quartPred),'+'}, {torch.cat(testData,testLabels), '+'})
gnuplot.plot({torch.cat(testData,LSPred),'+'}, {torch.cat(testData,testLabels), '+'})
-- predictions in blue, actual values in green