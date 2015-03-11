--****************************MACHINE LEARNING 2015 PRACTICAL ASSIGNMENT 5*************************--
--***********************************TRINITY TERM 2015*******WEEK 7********************************--

--commands written into Torch:

th> require 'nngraph'					  --  NETWORK:                                        OUTPUT
th> x1 = nn.Identity()()				  --      -->--[x1]--->--->--->--->--->---->----           ^ 
th> x2 = nn.Identity()()				  --      -->--[x2]--->--->--->-->--            \    ____  |
th> x3 = nn.Identity()()				  --                                \            \->|plus|-+
th> plus = nn.CAddTable()({x1,x2})		  --		   		                 >-->--[cmul]-->|____|
th> linear = nn.Linear(10,10)(x3)		  --                                /
th> cmul = nn.CMulTable()({x2,linear})	  --      ->--[x3]-->--[linear]-->--  
th> plus = nn.CAddTable()({cmul,x1})      --                  
th> network = nn.gModule({x1,x2,x3},{plus})

th> t1 = torch.randn(10)
--                {0.2165, 1.9396, 0.1998, 0.9769, 0.4812,-0.2308, 0.8400, 2.6602,-1.2385, 0.5235}
th> t2 = torch.randn(10)
--                {0.8252,-0.3970,-0.4846, 1.2337,-1.3375, 0.3837,-0.3003,-0.9580,-0.1301,-1.0483}
th> t3 = torch.randn(10)
--				  {-2.4137,-1.1654, 0.1266, 0.2670,-0.9744,-0.0243,-0.1900, 1.3731,0.6577,-0.8526}

th> network:forward({t1,t2,t3}) 
-- 				  {-0.4273, 1.8532,0.5345, 1.9886, 2.4574,-0.6712, 0.8907, 3.1905,-1.2368,-0.3677}	

-- code to check result:
th> params  = linear.data.module:parameters()[1]
th> bias    = linear.data.module:parameters()[2]
th> torch.cmul(t2,(bias+params * t3))+t1			--it works correctly; hurray!
-- 				  {-0.4273, 1.8532,0.5345, 1.9886, 2.4574,-0.6712, 0.8907, 3.1905,-1.2368,-0.3677}	

-- REMARK: putting x3->linear in as a separate module to use its output in checking that the output
-- of the whole network is correct failed miserably; arguably due to bugs in nngraph not allowing
-- connecting this module as a part of the big network.

--*************************************************************************************************--

--***Challenge 1: implementing LSTM****************************************************************--

function LSTM(xt,c_prev,h_prev) --implementing for same fixed size, to keep things easy.
	local inputSum = nn.CAddTable()({nn.Linear(10,10)(xt), nn.Linear(10,10)(h_prev)}) --sum input
	--define gates                                   (!omitting previous cell content!)  to gates
	local input_gate  = nn.Sigmoid()(inputSum) --equation 7 in http://arxiv.org/abs/1308.0850v5.pdf
	local forget_gate = nn.Sigmoid()(inputSum) --equation 8 in the above paper  | note that these
	local output_gate = nn.Sigmoid()(inputSum) --equation 10 in the above paper | sigmoids are all
										     --                                 | different units, so
										     -- note everything is a graph      | they have different
										     --           vertex in nngraph     | weight matrices
	local input_gate2 = nn.Tanh()(inputSum)  --this and the cell are equation 9 in the paper
	local cell        = nn.CAddTable()({nn.CMulTable()({forget_gate,c_prev}), 
									 nn.CMulTable()({input_gate,input_gate2})})
	return cell, nn.CMulTable()({output_gate, nn.Tanh()(cell)})
end

-- testing code for t1,t2,t3,x1,x2,x3 as defined above:
mem = LSTM(x1,x2,x3)									--NOTE THAT I AM ONLY TESTING NETWORK
network = nn.gModule({x1,x2,x3},{mem})					--THROUGHPUT, NOT CORRECT BEHAVIOUR!
print(network:forward({t1,t2,t3})) --prints a tensor of dimension 10, so the network is working.

--***Challenge 2: computation graph structure******************************************************--

-- Question: the graph described to nngraph is a computation graph, illustrating dependencies between
-- computations in a forward pass. By flipping the directions of the edges in this graph, we get the
-- backwards graph. Given a computation graph, which is always a directed acyclic graph, state an
-- algorithm that gives a valid sequence of computations as a list specifying an evaluation order for
-- this forward pass; we can just reverse this list to get an execution order for the backward pass.

-- Solution: the algorithm is just simple topological sort, which lists the entries in order of the
-- level they are from, starting from a given level. Its complexity is O(|V|+|E|), i.e. it is linear
-- in terms of the sum of the numbers of vertices and edges; reversing the output will therefore be
-- a valid order to visit the vertices for backpropagation.

