%Created by Mengdi Wang on 13-6-5.
%Copyright (c) 2013年 Mengdi Wang. All rights reserved.

%================viterbi function for HMM================
%Parameters: 
%	Pi: Initial state distribution
%	A: State-transition probability matrix
%	B: Observation probability distribution
% 	O: Observation sequence
%Return: 
%	q: state sequence
%	prob: MLE probability
function [q, prob] = viterbi(pi, A, B, O)
	
	%Variables definition
	N = size(A,1);
	T = size(O,2);
	prob = 0;
	q = ones(1,T);
	delta = zeros(T, N);
	psi = zeros(T, N);
	
	%Initial:
    %	$\delta_1(i) = \pi_ib_i(o_1)$ \\
    %	$\psi_1(i) = 0$ \\
	for (i=1:N)
		delta(1, i) = pi(i) * B(i, O(1));
		psi(1, i) = 0;
	end

    %Recursion:
    %	$\delta_t(j) = \max\limits_{1\leq i\leq N} [\delta_{t-1}(i)a_{ij}]b_j(o_t)$ \\
    %	$\psi_t(j) = \arg \max\limits_{1\leq i\leq N} [\delta_{t-1}(i)a_{ij}]$ \\
    %	$1 \leq t \leq T-1, 0 \leq j \leq N-1$ \\
	for (t=2:T)
		for (j=1:N)
			maxval = 0.0;
			maxidx = 1;
			for (i=1:N)
				val = delta(t-1, i) * A(i, j);
				if(val > maxval)
					maxval = val;
					maxidx = i;
				end
			end
			delta(t,j) = maxval * B(j, O(t));
			psi(t,j) = maxidx;
		end
	end
	
	%Termination:
	% $P^\ast = \max\limits_{1\leq i \leq N} [\delta_T(i)]$\\
	% $q^\ast_T = \arg \max\limits_{1\leq i\leq N} [\delta_T(i)]$\\
	for (i=1:N)
		if(delta(T, i) > prob)
			prob = delta(T, i);
			q(1, T) = i;	
		end
	end
	
	%Found Path
	% $q^\ast_t = \psi_{t+1}(q^\ast_{t+1})$ \\
	for (t=T-1:-1:1)
		q(1,t) = psi(t+1, q(1, t+1));
	end
end
%end of the file