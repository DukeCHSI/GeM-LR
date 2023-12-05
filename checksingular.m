%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
%                                                                 %
% Function: Check whether covariance matrix is ill-conditioned    %
% or nearly singular. If so, regularize covariance by changing    %
% the diagonal elements.                                          %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%

function [sigmanew]=checksingular(sigma, Xvar, shrinkrate);
dim=size(sigma,1);
sigmanew=sigma;


thred=mean(Xvar)*1.0e-4;
t=mean(diag(sigma));


% Guard against nearly singular matrix
if (rcond(sigma)<1.0e-8 | t<thred)
v1=sum(diag(sigma))/dim;
for i=1:dim
sigma(i,i)=sigma(i,i)+shrinkrate*v1;
end;
else
return;
end;
sigmanew=sigma;


% Check again
t=mean(diag(sigma));

if (rcond(sigma)<1.0e-8 | t<thred)
v1=mean(Xvar);
for i=1:dim
sigma(i,i)=sigma(i,i)+shrinkrate*v1;
end;
else
return;
end;
sigmanew=sigma;


t=mean(diag(sigma));

if (rcond(sigma)<1.0e-8 | t<thred)
fprintf('Warning: unable to solve singular covariance matrix, recommend to modify data\n');
sigma
error('checksingular: failed to modify sigma to be away from singular\n');
end;