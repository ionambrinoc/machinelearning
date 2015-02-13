--****************************MACHINE LEARNING 2015 PRACTICAL ASSIGNMENT 2***************************--
--***********************************TRINITY TERM 2015*******WEEK 3**********************************--


--****CODE - example-linear-regression.lua***********************************************************--
require 'torch'
require 'optim'
require 'nn'
										-- training data
data = torch.Tensor{{40,6,4},
					{44,10,4},
					{46,12,5},
					{48,14,7},
					{52,16,9},
					{58,18,12},
					{60,22,14},
					{68,24,20},
					{74,26,21},
					{80,32,24}} --{corn,fertilizer,insecticide}

model = nn.Sequential()                 -- define the container
ninputs = 2; noutputs = 1 				-- neuron, 2 inputs, 1 output
model:add(nn.Linear(ninputs, noutputs)) -- define the only module
criterion = nn.MSECriterion()			-- minimize mean squared error

x, dl_dx = model:getParameters() -- x = parameter vector; dl/dx = grads
feval = function(x_new) --eval loss function & grad
   if x ~= x_new then
      x:copy(x_new)
   end

   _nidx_ = (_nidx_ or 0) + 1 		   -- select new training sample - a row of "data"
   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = data[_nidx_]
   local target = sample[{ {1} }]      -- this funny looking syntax allows
   local inputs = sample[{ {2,3} }]    -- slicing of arrays.

   dl_dx:zero() -- reset gradients (otherwise always accumulated, to accomodate batch methods)

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))
	-- update gradient then accumulate parameters; backpropagation  
   return loss_x, dl_dx				   -- return loss(x) and dloss/dx (updated by backpropagation)
end

sgd_params = {	--train model using SGD
   weightDecay = 0, learningRate = 1e-3,   --10^(-3); decay regularizes solution / L2
   momentum = 0, learningRateDecay = 1e-4, --10^(-4); momentum averages steps over time
}

for i = 1,1e3 do 						   --number of epochs/full loops over training data
   current_loss = 0						   --estimate average loss
   for i = 1,(#data)[1] do 				   --loop over training data
    
      params,fs = optim.sgd(feval,x,sgd_params)  -- use optim's SGD algorithm, returns new x and the
												 -- value of loss functions at all points used by
      current_loss = current_loss + fs[1]		 -- algorithm.
   end
   current_loss = current_loss / (#data)[1]      -- report average error on epoch
   print('current loss = ' .. current_loss)
end

text = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}
print('id  approx   text')						 -- testing trained model
for i = 1,(#data)[1] do
   local myPrediction = model:forward(data[i][{{2,3}}])
   print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
end


--************** HAND-IN ***************************************************************************--
-- Question 2: testing the trained model on new set:	   -- predictions:
dataTest = torch.Tensor{ {6,4},{10,5},{14,8}}			   -- #   1e4 epochs 1e3 epochs 1e5 epochs
print('parameters:')	-- print table head				   -- 1    40.10       33.37       40.33
print(params)											   -- 2    43.88       40.44       44.03
														   -- 3    49.89       47.06       49.96
print('id test')		-- predict and print test values   -- parameters: bias insecticide fertilizer
for i=1,(#dataTest)[1] do 								   -- 1e3 epochs: 23.21   1.8392     -0.24
   local prediction = model:forward(dataTest[i][{{1,2}}])  -- 1e4 epochs: 31.63   1.1146     0.6676
   print(string.format("%2d  %6.2f", i, prediction[1]))    -- 1e5 epochs: 31.98   1.11       0.64
end
--**************************************************************************************************--
--Question 3: least squares solution - no optimization here, just using formula & data provided

y = data:narrow(2,1,1) --slice first column, then rest of it, and construct biased data Tensor
X = torch.cat(torch.ones((#data)[1],1),data:narrow(2,2,2))   -- by adding bias column (ones)
XT= X:transpose(1,2)										 -- swap width (1) and height (2)

biasedDataTest = torch.cat(torch.ones((#dataTest)[1],1), dataTest)
print(biasedDataTest)			-- also add bias column to test data        PREDICTION OUTPUTS:
																		    --       1. 40.32
print('LSParams')				-- print out parameters 					--       2. 44.03
LSParams = (torch.inverse(XT * X))*XT*y										--       3. 49.96
print(LSParams)
								-- predict by tensor multiplication
LSPredictions = biasedDataTest * LSParams							-- bias insecticide fertilizer
print(LSPredictions)			-- resulting parameters: 			  31.98     1.11      0.65

-- note convergence as number of epochs increases to the prediction given by the least squares model
--**************************************************************************************************--
