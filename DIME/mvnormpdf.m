function pdf = mvnormpdf(x,mu,Sigma)
% x  = p.n array of n values of p-dim MV normal
% mu = column p vector mean
% Sigma = p.p variance matrix 
% pdf = n vector of pdf values
%
[p n]=size(x); C=chol(Sigma); e=inv(C)'*(x-repmat(mu,1,n)); 
pdf = exp(-sum(e.*e)/2)/( prod(diag(C))*(2*pi)^(p/2) );
