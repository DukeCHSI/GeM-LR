%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
%                                                                 %
% Function: Convert the model parameters stored in stacked format %
% into the usual separated format, a row vector (numcmp) is prior %
% mu (dim, numcmp) is GMM means,                                  %
% sigma is GMM covariance, (dim,dim,numcmp)                       %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%

function [numcmp,a,mu,sigma]=GMMFormatConvert(dim,c);
% Convert storage format for Gaussian mixture model
numcmp=length(c.w);
a=c.w;
mu=c.supp(1:dim,:);
sigma=reshape(c.supp(dim+1:dim+dim*dim,:), dim,dim,numcmp);
