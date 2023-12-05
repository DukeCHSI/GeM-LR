%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
% Function: classify based on a given MLM model                   %
%-----------------------------------------------------------------%

function [pyi, pij]=MLMclassify(a,mu,sigma,beta, X, Xlogit)

[numdata,dim]=size(X);
dimlogit=size(Xlogit,2);
numcmp=length(a);

for j=1:numcmp
sigmainv(:,:,j)=inv(sigma(:,:,j));
sigmadetsqrt(j)=sqrt(det(sigma(:,:,j)));
end;

pij=zeros(numdata, numcmp);
pyij=zeros(numdata, numcmp);
pyi=zeros(1,numdata);
loglike=0.0;
for i=1:numdata
  tmp=0.0;
for j=1:numcmp
  pij(i,j)=a(j)/sigmadetsqrt(j)*exp(-0.5*(X(i,:)'-mu(:,j))'*sigmainv(:,:,j)*(X(i,:)'-mu(:,j)));
  
    if (pij(i,j)>=0)
tmp=tmp+pij(i,j);
  else
      fprintf('Warning: numerical error when computing pij: pij(%d, %d)=%f\n', i,j,pij(i,j));
      fprintf('sigmadetsqrt(%d)=%f\n',j, sigmadetsqrt(j));
      fprintf('Data point %d:\n',i);
      X(i,:)
      error('MLMclassify: Joint density of X and the component should be nonnegative\n');
  end;

end %% j=1:numcmp
for j=1:numcmp
if(tmp > 0)
     pij(i,j)=pij(i,j)/tmp;
    else
        pij(i,j) = 1/numcmp;
    end
end %% j=1:numcmp


for j=1:numcmp
  v1=exp(Xlogit(i,:)*beta(2:dimlogit+1,j)+beta(1,j));
  if (v1>1.0e+10)
    pyij(i,j)=1;
  else
    pyij(i,j)=v1/(1.0+v1); % class probability based on logistic model
  end;
end;

pyi(i)=sum(pij(i,:).*pyij(i,:));

end %% i=1:numdata

