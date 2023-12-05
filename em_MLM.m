%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%

%-------------------------------------------------------%
%- This subroutine estimates a gaussian mixture using  -%
%- EM and cluster data (both soft and hard) based on   -%
%- the mixture model.                                  -%
%- The input data X has every sample stored in a row   -%
%-------------------------------------------------------%
%-------------------------------------------------------%
% Output:                                               %
% mu: mean vectors of Gaussian components               %
% sigma: covariance matrices for Gaussian components    %
% a: prior probabilities of Gaussian components         %
% clsres: stores the identity of the most likely        %
%       cluster for each sample (hard classification)   %
% postp: stores the posterior probabilities of each     %
%        cluster for each sample. (soft classification) %
%-------------------------------------------------------%


function [c, beta, Wi, loglike,loglikepen]=em_MLM(X, Xlogit, Y, cinit, betainit, MLMoption);
% c.supp(dim+dim*dim,numcomponents) stores the mu and sigma of all components as an array.
% c.w stores the prior for each component
% beta stores the coefficient for the logistic regression model
% beta(dim+1,numcmp), each column contains coefficients for one logistic model
%%% cmcvflag=1, assume common covariance matrix within one cluster
% The prior distribution cpriorw(1, number of components) stores the componenent weights in
% the prior GMM.
% The prior distribution cpriorw(dim+dim*dim,number of components) stores the mean and
% covariance matrix for the Gaussian components in the prior GMM
% and the prior model.
% lambdaLasso(1,numcmp) a row vector that specifies a lambda for Lasso for each component
% separately
% Wi(1:numdata) is the instance weight to be optimized
% MLMoption.kappa=0 corresponds to NOT weighted case

%------------------------------%
% Estimate                     %
%------------------------------%
kappa=MLMoption.kappa;

[numdata,dim]=size(X);
dimlogit=size(Xlogit,2);
numcmp=length(cinit.w);

lambdaLasso=zeros(1,numcmp);
m=length(MLMoption.lambdaLasso);
k=1;
for i=1:numcmp
lambdaLasso(i)=MLMoption.lambdaLasso(k);
k=k+1;
if (k>m) k=1; end;
end;

% initialize the weight to uniform
% Note that Wi is normalized to have sum equal to numdata, Not 1
Wi=ones(1,numdata); 

% Sanity check for input X
Xvar=var(X);
if (mean(Xvar)<1.0e-6)
    fprintf('Warning: average variance is very small: %f, may lead to singular matrix\n', mean(Xvar));
end;
for j=1:dim
    if (Xvar(j)<1.0e-6)
        fprintf('Warning: variance of dimension %d is very small: %f, may lead to singular matrix\n', j, Xvar(j));
    end;
end;

X=X'; %%% !!! Transposed input matrix from now on
Xlogit=Xlogit';
mu=cinit.supp(1:dim,:);
sigma=reshape(cinit.supp(dim+1:dim+dim*dim,:),dim,dim,numcmp);
a=cinit.w;
beta=betainit;

minloop=max([MLMoption.minloop,2]);
maxloop=max([MLMoption.maxloop,5]);

oldloglike=-1.0e+30;
oldloglikepen=oldloglike;

for j=1:numcmp
sigmainv(:,:,j)=inv(sigma(:,:,j));
sigmadetsqrt(j)=sqrt(det(sigma(:,:,j)));
end;

loop=1;
while (loop<maxloop)
%%%% Compute the posterior probabilities 
%loop

pij=zeros(numdata, numcmp);
loglike=0.0;
for i=1:numdata
  tmp=0.0;
for j=1:numcmp
  v1=exp(beta(2:dimlogit+1,j)'*Xlogit(:,i)+beta(1,j));
  if (v1>1.0e+10)
    pyij(i,j)=1;
  else
    pyij(i,j)=v1/(1.0+v1); % class probability based on logistic model
  end;
    
    if (MLMoption.Yalpha==0)
        pij(i,j)=a(j)/sigmadetsqrt(j)*exp(-0.5*(X(:,i)-mu(:,j))'*sigmainv(:,:,j)*(X(:,i)-mu(:,j)))*((Y(i)*pyij(i,j)+(1-Y(i))*(1-pyij(i,j)))^MLMoption.Ypower);
    else
        pij(i,j)=a(j)/sigmadetsqrt(j)*exp(-0.5*(X(:,i)-mu(:,j))'*sigmainv(:,:,j)*(X(:,i)-mu(:,j)));
        v5=Y(i)*pyij(i,j)+(1-Y(i))*(1-pyij(i,j));
        v5=exp(MLMoption.Yalpha*(v5-0.5))/(1.0+exp(MLMoption.Yalpha*(v5-0.5)));
        pij(i,j)=pij(i,j)*v5;
    end;
    
  if (pij(i,j)>=0)
tmp=tmp+pij(i,j);
  else
      fprintf('Warning: numerical error when computing pij: pij(%d, %d)=%f\n', i,j,pij(i,j));
      fprintf('sigmadetsqrt(%d)=%f\n',j, sigmadetsqrt(j));
      fprintf('beta(:,%d)\n',j);
      beta(:,j)
      fprintf('pyij(%d, %d)=%f\n',i,j,pyij(i,j));
      fprintf('Data point %d:\n',i);
      X(:,i)'
      error('em_MLM: joint density of X, Y, and component should be nonnegative');
  end;

end %% j=1:numcmp
for j=1:numcmp
if(tmp > 0)
     pij(i,j)=pij(i,j)/tmp;
    else
        pij(i,j) = 1/numcmp;
    end
end %% j=1:numcmp

%%!! sanity check !!%%%%%
for j=1:numcmp
  if (~(pij(i,j)>=0 | pij(i,j)<0)) % NaN appears
    fprintf('pij(%d, %d)=%f\n',i,j,pij(i,j));
    error('em_MLM: Numerical error with computing posterior pij');
  end;
end;


loglike=loglike+(log(tmp)-dim/2*log(2*pi))*Wi(i); % weighted sum of log likelihood
end %% i=1:numdata

% Take into account the weights assigned to each sample point
pij_wt=pij;
for i=1:numdata
    pij_wt(i,:)=pij(i,:)*Wi(i); % normalize so that on average each instance has weight 1
end;

% Compute penalizede loglikelihood with MAW distance to prior as the penalty
if (kappa>0)
Wiunit=Wi/sum(Wi);
Wientropy=0.0; % actually the negative of the entropy
for i=1:numdata
  if (Wiunit(i)>0)
    Wientropy=Wientropy+Wiunit(i)*log(Wiunit(i));
  end;
end;
else
  Wientropy=0;
end;

if (MLMoption.algorithm==1)
    penbeta=MLMoption.AlphaLasso*sum(abs(beta(2:dimlogit+1,:)))+(1-MLMoption.AlphaLasso)*sum(beta(2:dimlogit+1,:).^2);
loglikepen=loglike-sum(lambdaLasso.*penbeta)-kappa*Wientropy; 
else
    loglikepen=loglike-kappa*Wientropy; 
end;

if (abs((loglikepen-oldloglikepen)/oldloglikepen) < MLMoption.stopratio & loop>minloop)
break;
end;

if (loglikepen<oldloglikepen & loop>minloop)
loop
[loglike, oldloglike, loglikepen,oldloglikepen]
break;
end;

oldloglike=loglike;
oldloglikepen=loglikepen;

%%%%% Start maximization step 

%%%% Optimize a
pj=sum(pij_wt);
a=pj/sum(pj);
numnonzero=sum(~(a>0));
if (numnonzero==numcmp)
  fprintf('All components have non-positive prior\n');
  fprintf('pj:\n');
  pj
  fprintf('a:\n');
  a
end;

%% Before updating mu, sigma, and beta, store the result from previous round as we may keep them in special cases
muprev=mu;
sigmaprev=sigma;
betaprev=beta;

%%% Optimize mu
mu=X*pij_wt;
for j=1:numcmp
if (pj(j)>0)
mu(:,j)=mu(:,j)/pj(j);
else
mu(:,j)=muprev(:,j);
fprintf('Warning: zero prior for component %d, Component mean, Covariance, and Beta all set to the same as previous round.\n',j);
end;
end;

%%%% Optimize sigma
for j=1:numcmp
Phi=zeros(dim,dim);
for i=1:numdata
Phi=Phi+pij_wt(i,j)*(X(:,i)-mu(:,j))*(X(:,i)-mu(:,j))';
end;
if (pj(j)>0)
sigma(:,:,j)=Phi/sum(pj(j));
else
sigma(:,:,j)=sigmaprev(:,:,j);
end;

% Guard against nearly singular matrix
sigma(:,:,j)=checksingular(sigma(:,:,j), Xvar, 0.05);

end %%% j=1:numcmp

% Revise the sigma matrix to fit the regularization constraints
sigma=ConstrainSigma(a, sigma,dim,numcmp, Xvar, MLMoption);
  

%% Perform Lasso for Y
for j=1:numcmp
    if (pj(j)>0)
        if (pj(j)>=3)
            if (MLMoption.algorithm==1) 
        [beta_nointercept(:,j),STATS]=lassoglm(Xlogit',Y,MLMoption.DISTR,'Weights',pij_wt(:,j),'Lambda',[lambdaLasso(j)], 'Alpha', MLMoption.AlphaLasso);
        beta(:,j)=[STATS.Intercept; beta_nointercept(:,j)];
        % compute beta(:,j) for original data (not standardized)
            else
                beta(:,j)=glmfit(Xlogit',Y,MLMoption.DISTR,'link','logit','Weights',pij_wt(:,j)/pj(j));
            end;
        else % if (pj(j)>=3
             beta(:,j)=glmIntercept(Y,pij_wt(:,j),dimlogit,MLMoption.DISTR);
        end; % if (pj(j)>=3)
    else % if (pj(j)>0)
        beta(:,j)=betaprev(:,j);
    end;
end;


%% Update weights for data points Wi

%sigmainv and sigmadetsqrt will be used in the next loop
for j=1:numcmp
sigmainv(:,:,j)=inv(sigma(:,:,j));
sigmadetsqrt(j)=sqrt(det(sigma(:,:,j)));
end;

if (kappa>0)
Li=zeros(1,numdata);
for i=1:numdata
for j=1:numcmp
  v1=beta(2:dimlogit+1,j)'*Xlogit(:,i)+beta(1,j);
  if (exp(v1)==Inf)
    v2=-v1+Y(i)*v1+log(a(j))-log(sigmadetsqrt(j))-0.5*(X(:,i)-mu(:,j))'*sigmainv(:,:,j)*(X(:,i)-mu(:,j));
  else
    v2=-log(1+exp(v1))+Y(i)*v1+log(a(j))-log(sigmadetsqrt(j))-0.5*(X(:,i)-mu(:,j))'*sigmainv(:,:,j)*(X(:,i)-mu(:,j));
  end;
  if (a(j)>0)
    Li(i)=Li(i)+v2*pij(i,j);
  end;
  end %% j=1:numcmp
end %% i=1:numdata

Wi=exp(Li/kappa);
if (sum(Wi)==0)
Wi=ones(1,numdata); % sum(Wi) normalized to numdata, Not 1
else
        Wi=Wi/sum(Wi)*numdata; % sum(Wi) normalized to numdata, Not 1
end;

%%!! sanity check !!%%%%%
for i=1:numdata
  if (~(Wi(i)>=0 | Wi(i)<0)) % NaN appears
    fprintf('Wi(%d)=%f\n',i,Wi(i));
    error('em_MLM: Numerical error with computing weights Wi');
  end;
end;

end;

%loop 
%loglike

loop=loop+1; % If to view the intermediate results, remove ";" at the end
loglikepen;

end %%% while (loop<maxloop)

%% Convert back to the format
c.w=a;
c.supp(1:dim,:)=mu;
for i=1:numcmp
  c.supp(dim+1:dim+dim*dim,i)=reshape(sigma(:,:,i),dim*dim,1);
end;


