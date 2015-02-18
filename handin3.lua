--****************************MACHINE LEARNING 2015 PRACTICAL ASSIGNMENT 2***************************--
--***********************************TRINITY TERM 2015*******WEEK 3**********************************--

-- Introduction and review: as in the linear regression example, this feval function takes a data
-- point and computes the loss function (f itself) at said point, as well as f's derivative
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
 -- (...) -- 				 
  local out = model:forward(test.data)		-- now evaluate test set performance
  local currentLoss = criterion:forward(out, test.labels)
  test_losses[#test_losses+1] = currentLoss	--epoch test loss
  print("epoch " .. epoch .. ', full epoch loss '.. epoch_loss .. ', test loss ' ..currentLoss)
end

losses[#losses+1] = epoch_loss/(train.data:size(1)/ opt.batch_size)        --scale down for plotting
gnuplot.plot({torch.range(1, #losses),torch.Tensor(losses),'-'},              --plots both curves on
             {torch.range(1, #test_losses),torch.Tensor(test_losses),'~'})    --same image

-- here, training error converges downwards towards 0, but test error is massive; obvious overfitting.

-- Better optimization:

--++Running LBFGS with minibatch size 500 on training set of 40000 gives some of the best results I 
--have seen; it is pretty quick to converge (~7 epochs), to a training error of ~300 & test error 
--~3.12; going up to 15 epochs does not help too much, since convergence has been reached.
--++Even better results are obtained using LBFGS on 12 epochs on all the training data, with a batch
--size of 650; this whittles training error down to ~221 and test error to ~2.30. Based on these
--results, one would seem to think that the optimal batch size for LBFGS is around 1.1% of the 
--training set; with batch sizes that are too small, test loss is very large; if batch size is more
--than this, one can notice wildly oscillating test errors after a number of methods; note that as  
--a 2nd order method, LBFGS is pretty quick to process and converge. Note that we are lucky here,
--since we have a simple model where LBFGS finds the extremum quickly; it is often the case that 
--it is slow to converge. 

--++Running Adagrad on the whole training set seems slightly slower to process and certainly slower
--to converge; a batch size of 100 on 20 epochs reduces trest error to 4.75 (with larger training
--error) than LBFGS; increasing batch size to 200 gives higher test error (less training error);
--larger minibatches only reduce training error. So adagrad seems to overfit the training data with
--larger minibatches; smaller minibatches seem to give lower test error. The best results I obtained
--with adagrad on the whole test set are with a minibatch size of 5, for 30 epochs; test loss never
--became less than 3.70, however (strangely, running time increases with batch size decrease).

--++SGD is hard to configure; its test error oscillates wildly between epochs (see fourth image),
--even though it is very quick to evaluate epochs; convergence is slow, and requires very large
--batch sizes (not a big problem, since SGD goes through them quickly); it seems to slow down as
--the batch size becomes small, and needs a large number of epochs to give relts that are not very
--wrong; even so, I have not been able to reduce test loss with SGD to less than 800.

--++Given the choice here, considering convergence speed, accuracy and execution time, I would 
--optimize with LBFGS on 12 epochs with a batch size of 1.1% of the training set size; the code
--given above allows us to observe test/classification error; the larger batch size required by
--LBFGS is a consequence of its poor resillience to noise.

-- GRAPHS: 1. SGD, batch size 500, 100 epochs		4. Adagrad, batch size 200, 20 epochs
--	   2. Adagrad, batch size 5, 30 epochs		5. LBFGS, batch size 650, 12 epochs (full set)
--	   3. Adagrad, batch size 100, 20 epochs	6. LBFGS, batch size 500, 10 epochs (40k set)
