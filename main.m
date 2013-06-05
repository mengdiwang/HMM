%Created by Mengdi Wang on 13-6-5.
%Copyright (c) 2013年 Mengdi Wang. All rights reserved.

%=================Test function for HMM==================
clear; close all; clc;

%test start
fprintf('HMM Decoding problem test\n');
%variables 
M 	= 2;
N 	= 3;
T 	= 10;
A 	= [0.9,0.05,0.05; 0.45,0.1,0.45; 0.45,0.45,0.1];
A2  = [0.333 0.333 0.333; 0.333 0.333 0.333; 0.333 0.333 0.333;];
B 	= [0.5,0.5; 0.75,0.25; 0.25,0.75];
pi 	= [0.333,0.333,0.333];
O 	= [1,1,1,1,2,1,2,2,2,2];

%call function
[q, prob] = viterbi(pi, A2, B, O);

%print results
fprintf('log prob: %f\n', log(prob));
fprintf('state sequence:\n');
for(i=1:size(O,2))
	fprintf('%d ', q(1,i));
end
fprintf('\n');

fprintf('HMM Learning problem test\n');
%test learning problem
[A2,B,pi]=EMHMM(pi, A2, B, O);
A2
B
pi



