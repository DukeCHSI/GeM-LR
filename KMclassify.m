%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
% Function: classify based on a given MLM model                   %
%-----------------------------------------------------------------%

function [pyi, pij]=KMclassify(a,mu,sigma,beta, X, Xlogit)

[numdata,dim]=size(X);
dimlogit=size(Xlogit,2);
numcmp=length(a);

dij=zeros(numdata, numcmp);
pyi=zeros(1,numdata);
pij=zeros(numdata, numcmp);
loglike=0.0;

for i=1:numdata
for j=1:numcmp
    dij(i,j)=sum((X(i,:)'-mu(:,j)).^2);
end;

[mv,jj]=min(dij(i,j));
pij(i,jj)=1; % The chosen component

% jj is the chosen component to perform logistic regression
v1=exp(Xlogit(i,:)*beta(2:dimlogit+1,jj)+beta(1,jj));
if (v1>1.0e+10)
  pyi(i)=1;
else
  pyi(i)=v1/(1.0+v1); % class probability based on logistic model
end;

end %% i=1:numdata

