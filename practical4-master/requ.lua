require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)			--just implemented formula in manual: (x>0)(*)x(*)x
  self.output = torch.Tensor()
  self.output:resizeAs(input):copy(input)
  self.output:cmul(torch.gt(input,0):double()):cmul(input)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)       --derivative: gradInput = gradOutput * dz/dx
  self.gradInput = torch.Tensor()					   --by chain rule, where z=0 if x<0 => dz/dx=0
  self.gradInput:resizeAs(gradOutput):copy(gradOutput) --otherwise dz/dx = 2x; note gt is strict.
  self.gradInput:cmul(torch.gt(input,0):double()):mul(2):cmul(input)
  return self.gradInput 							   --note that loops are not used.
end