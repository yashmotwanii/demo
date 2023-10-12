clc;clear all;close all;


% Define the problem parameters
L = 384; % number of rows in Z and B
M = 24; % number of columns in Z and X
N = 16; % number of columns in B and rows in X

% Generate random complex matrices for Z, B, and X
B = randn(L, N)+ 1i*randn(L, N);
W = 1*randn(L,M);

j = [4 2];
i = [12 1];
v = [0.8+0.9i 1.8+0.2i];

S = sparse(i,j,v,N,M);
Z = B*S+W;

%% VECTORIZE

Zvec=Z(:);
I=eye(M);
Bvec=kron(transpose(I),B);
% [lambda,S_lasso]=lassoAlgo(Bvec,Zvec); 
[lam,A,S_clars] = clarswlasso(Zvec,Bvec,0,0);
% A
S_clars = reshape(S_clars, N, M);
% S_lasso = reshape(S_lasso, N, M);
% for i=1:64
%     clars = reshape(B1(:,), N, M);
%     err(i)=norm(S-clars)^2;
% end
% S_lasso  = zeros(N,M);
% lambda =  10;

%% Solve columns independently
% 
% for j=1:M
%     j
% %     [lambda,S_lasso(:,j)]=lassoAlgo(B,Z(:,j));
%     [~,A,S_clars(:,j),B1]= clarswlasso(Z(:,j),B);
% end
% 


%% Error
S_ls = pinv(B)*Z;
% disp("Error_lasso: " + norm(S-S_lasso)^2);
disp("clars: " + norm(S-S_clars)^2) ;
disp("Error_ls: " + norm(S-S_ls)^2) ;


