--****************************MACHINE LEARNING 2015 PRACTICAL ASSIGNMENT 2***************************--
--***********************************TRINITY TERM 2015*******WEEK 3**********************************--

-- task 1: introduction and review: as in the linear regression example, this feval function takes
-- a data point and computes the loss function (f itself) at said point, as well as f's derivative
-- (gradient) at the same point.

-- to change initialization point, we change the integer in the line:
local x = torch.Tensor{5} -- finds local minimum 3.679312 (very close to inflection point, i.e. H=0)
local x = torch.Tensor{2} -- finds local minimum 0.001666 (very colse to actual local minimum)
local x = torch.Tensor{1} -- same result as above; note that gradient is 0 here, but hessian is not
local x = torch.Tensor{-2}-- gives -0.00166, due to symmetry of the (even) function; the procedure,
			  -- however, seems to be an odd function
local x = torch.Tensor{-1}-- as expected, same result.
local x = torch.Tensor{-5}-- as expected, finds -3.67
local x = torch.Tensor{0} -- finds actual minimum 0; lucky shot
local x = torch.Tensor{10}-- outlying example gives 9.636460; actually maximum, gradient is 0

-- these have gradients calculated on a single data point, and not on mini-batches.

--**************************************************************************************************--
--************** HAND-IN ***************************************************************************--

-- Code to evaluate test set performance after every epoch: requires modifying epoch loop
for epoch = 1, epochs do
  epoch_loss = 0

  -- ... --

  -- now evaluate test set performance
  epoch_test_loss = 0

  local out = model:forward(test.data)
  local currentLoss = criterion:forward(out, test.labels)
  epoch_test_loss=currentLoss

  test_losses[#test_losses+1] = epoch_test_loss
  print("epoch " .. epoch .. ', full epoch loss ' .. epoch_loss .. ', test loss ' .. epoch_test_loss)
end

gnuplot.plot({torch.range(1, #losses),torch.Tensor(losses),'-'},              --plots both curves on
             {torch.range(1, #test_losses),torch.Tensor(test_losses),'~'})    --same image
