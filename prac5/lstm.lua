require 'nngraph'

function LSTM(xt,c_prev,h_prev) --implementing for same fixed size, to keep things easy.
	local inputSum = nn.CAddTable()({nn.Linear(10,10)(xt), nn.Linear(10,10)(h_prev)}) --add up everything
	--define gates
	local input_gate  = nn.Sigmoid()(inputSum) --equation 7 in http://arxiv.org/abs/1308.0850v5.pdf
	local forget_gate = nn.Sigmoid()(inputSum) --equation 8 in the above paper  | note that these
	local output_gate = nn.Sigmoid()(inputSum) --equation 10 in the above paper | sigmoids are all
										     --                                 | different units, so
										     --                                 | they have different
										     --                                 | weight matrices
	local input_gate2 = nn.Tanh()(inputSum)  --this and the cell are equation 9 in the paper
	local cell        = nn.CAddTable()({nn.CMulTable()({forget_gate,c_prev}), 
									 nn.CMulTable()({input_gate,input_gate2})})
	return cell, nn.CMulTable()({output_gate, nn.Tanh()(cell)})
end



t1 = torch.Tensor({0.2165, 1.9396, 0.1998, 0.9769, 0.4812,-0.2308, 0.8400, 2.6602,-1.2385, 0.5235})
t2 = torch.Tensor({0.8252,-0.3970,-0.4846, 1.2337,-1.3375, 0.3837,-0.3003,-0.9580,-0.1301,-1.0483})
t3 = torch.Tensor({-2.4137,-1.1654, 0.1266, 0.2670,-0.9744,-0.0243,-0.1900, 1.3731,0.6577,-0.8526})

x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Identity()()
--mem = LSTM(x1,x2,x3)
--network = nn.gModule({x1,x2,x3},{mem})

--print(network:forward({t1,t2,t3}))