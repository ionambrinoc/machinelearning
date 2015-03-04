require 'requ'						-- NOTE: Assumes input and output to module are 1-dimensional, 
									-- i.e. doesn't test the module in mini-batch mode. 
local function jacobian_wrt_input(module, x, eps)

  local z = module:forward(x):clone()					--compute true Jacobian (rows=over outputs, 
  local jac = torch.DoubleTensor(z:size(1), x:size(1))  --cols = over inputs, as in practical manual)
  
  local one_hot = torch.zeros(z:size())					--get true Jacobian, ROW BY ROW
  for i = 1, z:size(1) do
    one_hot[i] = 1
    jac[i]:copy(module:backward(x, one_hot))
    one_hot[i] = 0
  end
  
  local jac_est = torch.DoubleTensor(z:size(1), x:size(1)) --finite-differences Jacobian, COL BY COL
  for i = 1, x:size(1) do
    x[i] = x[i] + eps
    local z_plus = torch.DoubleTensor(z:size(1))
    z_plus:copy(module:forward(x))
    x[i] = x[i] - 2*eps
    local z_minus = module:forward(x)   --copy because of buffered reader issue
    jac_est[{{},i}]:copy(z_minus:mul(-1)):add(1, z_plus):div(2*eps) -- change eps to 2*eps & z to z-
    x[i] = x[i] + eps								 --restore value of x
  end

  local abs_diff = (jac - jac_est):abs()             --compute (symmetric) relative error of gradient
  return jac, jac_est, torch.mean(abs_diff), torch.min(abs_diff), torch.max(abs_diff)
end

torch.manualSeed(1)		  -- now test layer in isolation
local requ = nn.ReQU()

local x = torch.randn(10) -- random input to layer
print(x)
print(jacobian_wrt_input(requ, x, 1e-6))		--mean of absolute difference:6.73*10^(-12); very low

