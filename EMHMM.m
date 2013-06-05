%Created by Mengdi Wang on 13-6-5.
%Copyright (c) 2013年 Mengdi Wang. All rights reserved.

%================Estimate Maximization function for HMM================
%Parameters:
%	pi: initial state distribution
%	A: state-transition probability matrix
%	B: Observation probability distribution matrix
%	O: observation sequence
%Return:
%	A: \hat{a} 
%	B: \hat{b}
%	pi: \hat{\pi} 
%Algorithm:
%	E step:
%	Forward and Backward algorithm
%	M step:
%	$\hat{a}_{i,j} = \frac{\sum \xi_t(i,j)}{\sum \gamma_t(i)}$ \\
%	$\hat{b}_j(k) = \frac{\sum_{t,o_t=k} \gamma_t(j)}{\sum_t \gamma(j)}$ \\
%
%	$1. initial model \lambda_0$
%	$2. compute \lambda based on \lambda_0 and observation O$
%	$3. if \log P(O|\lambda) - \log P(O|\lambda) < Delta stop$
%	$4. else set \lambda_0 \gets \lambda and go to step 2$

function [A,B,pi]=EMHMM(pi, A, B, O)
	DELTA = 0.001;
	probprev = 0;
	N = size(A, 1);
	T = size(O, 2);
	M = size(B, 2);

	while(1)
		
		[alpha, prob, scaleM] = Forward(T, N, pi, A, B, O, 1);
		[beta, pprob]		  = Backward(T, N, A, B, O, 1, scaleM);
		gamma				  = CalcGamma(T, N, alpha, beta);
		xi					  = CalcXi(T, N, A, B, O, alpha, beta);
		
		%Reestimate \pi
		for(i=1:N) 
			pi(i) = gamma(1,i);
		end
		
		%Reestimate A,B
		%	$\hat{a}_{i,j} = \frac{\sum \xi_t(i,j)}{\sum \gamma_t(i)}$ \\
		%	$\hat{b}_j(k) = \frac{\sum_{t,o_t=k} \gamma_t(j)}{\sum_t \gamma(j)}$ \\
		for(i=1:N)
			denomB = sum(gamma(:,i));
			denomA = denomB - gamma(T,i);
				
			for(j=1:N)
				numerA=0.0;
				for(t=1:T-1)
					numerA += xi(t,i,j);
				end
				%A(i,j) = .001+.999*numerA / denomA;
				A(i,j) = numerA / denomA;
			end
				
			for(k=1:M)
				numerB = 0.0;
				for(t=1:T)
					if(O(t)==k)
						numerB += gamma(t, i);
					end
				end
				%B(i, k) = .001+.999*numerB / denomB;
				B(i, k) = numerB / denomB;
			end
		end
	
		delta = abs(prob - probprev);
		probprev = prob;
		if(delta <= DELTA)
			break;
		end
	end
end
%==========================Calculate $\alpha$===========================
%Parameters:
%	T, N, pi, A, B, O, 
%	scale: if scale = 1, scale the calculation
%Return:
%	alpha: alpha vector
%	prob: MLE probability
%	scaleM: $\sum_{i=1}^{N}\alpha_t(i)$ to be used for scaling beta
%Algorithm:
%Not scale:
%	$\alpha_1(i) = \pi_ib_i(o_1)$\\
%	$\alpha_{t+1}(j) := [\sum_{i=1}^{N} \alpha_t(i)a_{i,j}]b_j(o_{t+1}), 1 \leq t\leq T-1$\\
%Scale:
%	$\hat{\alpha}_t(i) = \frac{\alpha_t(i)}{\sum_{i=1}^{N}\alpha_t(i)}$
function [alpha,prob,scaleM]=Forward(T, N, pi, A, B, O, scale)
	alpha = zeros(T, N);
	scaleM = zeros(T, 1);
	for(i=1:N)
		alpha(1, i) = pi(i) * B(i, O(1));
		scaleM(1) += alpha(1, i);
	end
	
	if(scale==1)
		alpha(1, :) /= scaleM(1);
	end
	for(t=1:T-1)
		for(j=1:N)
			suma = 0.0;
			for(i=1:N)
				suma += alpha(t, i) * A(i, j);
			end
			alpha(t+1, j) = suma * B(j, O(t+1));
			scaleM(t+1) += alpha(t+1, j);
		end
		if(scale==1)
			alpha(t+1, :) /= scaleM(t+1);
		end
	end
	
	prob = 0.0;
	if(scale==1)
		for (t = 1 : T)
			%(102) $log[P(O|\lambda)]= - \sum_{t=1}^{T}\log c_t$\\
			prob += log(scaleM(t));
		end
	else
		for(i=1:N)
			prob += alpha(T, i);
		end
	end
end
%==========================Calculate $\beta$===========================
%Parameters:
%	T, N, A, B, O, scale
%	scaleM: scale matrix calculated in Forward function
%Return:
%	beta: beta vector
%	prob: MLE probability
%Algorithm:
%Not scale
%	$\beta := \sum_{j=1}^{N} a_{i,j}b_j(o_{t+1})\beta_{t+1}(j), 1 \leq i \leq N, t = T-1,...,1 $ \\
%Scale
%	$\hat{\beta} = c_t\beta_t(i)$\\
%	$c_t = \frac{1}{\sum_{i=1}^{N}\alpha_t(i)}$\\
function [beta,prob]=Backward(T, N, A, B, O, scale, scaleM)
	beta = ones(T, N);
	
	if(scale==1)
		beta /= scaleM(T,1);
	end
	
	for(t=T-1:-1:1)
		for(i=1:N)
			sumb = 0.0;
			for(j=1:N)%use sum
				sumb += A(i,j) * B(j, O(t+1)) * beta(t+1, j);
			end
			beta(t, i) = sumb;
			if(scale==1)
				beta(t, i) /= scaleM(t,1);
			end
		end
	end
	
	v = beta(1,:);
	prob = sum(v);
end
%==========================Calculate $\xi$===========================
%Parameters:
%	T, N, A, B, O, 
%	alpha: result of Forward function
%	beta: result of Backward function
%Return:
%	xi: xi vector
%Algorithm:
%	$\xi_t(i,j) := \frac{\alpha_t(i)a_{i,j}b_j(o_{t+1})\beta_{t+1}(j)}
%					{\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_t(i)a_{i,j}b_j(o_{t+1})\beta_{t+1}(j)} $\\
function xi=CalcXi(T, N, A, B, O, alpha, beta)
	xi = repmat(0, [T N N]);
	
	for(t=1:T-1)
		sumx = 0.0;
		for(i=1:N)
			for(j=1:N)
				xi(t,i,j) = alpha(t, i) * beta(t+1, j) * A(i,j) * B(j,O(t+1));
				sumx += xi(t,i,j);
			end
		end
	
		xi(t,:,:) /= sumx;
	end
end
%==========================Calculate $\gamma$===========================
%Parameters:
%	T, N, alpha, beta
%Return:
%	gamma: gamma vector
%Algorithm:
%	$\gamma_t(i) := \frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^{N} \alpha_t(j)\beta_t(j)} $\\
function gamma=CalcGamma(T, N, alpha, beta)
	gamma=zeros(T,N);
	
	for(t=1:T)
		base = 0.0;
		for(j=1:N)
			gamma(t,j) = alpha(t,j) * beta(t,j);
			base += gamma(t, j);
		end
		
		gamma(t,:) /= base;
	end
end
%end of the file
