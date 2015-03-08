local create_model = require 'create_model'				-- loads reQU

-------------------------------------------------------------
local opt = { nonlinearity_type = 'requ' }				-- SETTINGS

-- function that numerically checks gradient of the loss: f is the scalar-valued function,
-- g returns the true gradient (assumes input to f is a 1d tensor)
local function checkgrad(f, g, x, eps)
  local grad = g(x)										-- compute true gradient
  
  local eps = eps or 1e-11								-- compute numeric approximations to gradient
  local grad_est = torch.DoubleTensor(grad:size())
  for i = 1, grad:size(1) do
    x[i]   = x[i] + eps
    fplus  = f(x)
    x[i]   = x[i] - 2*eps
    grad_est[i] = (fplus-f(x))/(2*eps)
    x[i]   = x[i] + eps									-- and reset x[i] to initial value.
  end
  
  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)
  return diff, grad, grad_est			-- return difference, true gradient, and estimated gradient
end				

function fakedata(n)
    local data = {}
    data.inputs = torch.randn(n, 4)                     -- random standard normal distro for inputs
    data.targets = torch.rand(n):mul(3):add(1):floor()  -- random integers from {1,2,3}
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
torch.manualSeed(1)
local data = fakedata(5)
local model, criterion = create_model(opt)
local parameters, gradParameters = model:getParameters()

local f = function(x)								    -- returns loss(params)
  if x ~= parameters then
    parameters:copy(x)
  end
  return criterion:forward(model:forward(data.inputs), data.targets)
end

local g = function(x)									-- returns dloss(params)/dparams
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  local outputs = model:forward(data.inputs)
  criterion:forward(outputs, data.targets)
  model:backward(data.inputs, criterion:backward(outputs, data.targets))

  return gradParameters
end
												-- returns 1+72761*10^(-13) for eps=1e-7,
local diff = checkgrad(f, g, parameters)		-- 1+728*10(-13) for eps=1e-9 & 1+7*10^(-13)	
print(diff)										-- for eps=1e-11; similar magnitude to epsilon

