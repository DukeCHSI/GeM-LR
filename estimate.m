%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
%                                                                 %
% Function: EM algorithm for estimating MLM                       %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
 
function [c, beta, rsigma2, loglike, loglikepen, loglikeY, Wi]=estimate(X, Xlogit, Y, MLMoption);

dim=size(X,2);
numdata=size(X,1);
numcmp=MLMoption.numcmp(1);

%==============================================
% Experiment with Mixture of Linear Models
%==============================================

[muinit, sigmainit, ainit,betainit,rsigma2]=initemMLM(X, Xlogit, Y, MLMoption);

%[a1,a2,t1,AUC]=perfcurve(Y, pyi,'1'); %get AUC
if (MLMoption.verbose>0) % Otherwise classification accuracy within training not reported
    
if (strcmp(MLMoption.DISTR,'binomial'))
[pyi,pij]=MLMclassify(ainit,muinit,sigmainit,betainit,X, Xlogit);
pyihard=pyi>0.5; %pyi is posterior of class 1
fprintf('Classification accuracy by initial model (within training): %f \n', sum(Y'==pyihard)/length(Y));
[xcord,ycord,T,AUC]=perfcurve(Y,pyi',1); % Area under curve
fprintf('Classification AUC by initial model (within training): %f\n', AUC);
else
  [pyi,pij]=MLMregress(ainit,muinit,sigmainit,betainit,X, Xlogit);
  v1=sum((pyi-Y').*(pyi-Y'))/numdata;
  fprintf('Regression MSE by initial model (within training): %f\n', v1);
end;

end;


%----- EM for MLM with MAW penalty ------
%% indmaw stores the clustering result
%% postp_maw stores the posterior probilities of each cluster
%% alpha is the penalty coefficient, need to experiment with its value
%% cpriorsupp stores the Gaussian component parameters, cpriorw cstores the weights of the components

cinit.w=ainit;
cinit.supp=zeros(dim+dim*dim,numcmp);
cinit.supp(1:dim,:)=muinit;
for k3=1:numcmp
	 cinit.supp(dim+1:dim+dim*dim,k3)=reshape(sigmainit(:,:,k3),dim*dim,1);
end;

if (MLMoption.NOEM==1)
    c=cinit;
    beta=betainit;
    loglike=0;
    loglikepen=0.0;
    Wi=ones(1,numdata); 
else
  if (strcmp(MLMoption.DISTR,'binomial'))
    [c, beta, Wi, loglike, loglikepen]=em_MLM(X,Xlogit, Y, cinit,betainit, MLMoption);
  else % regression case
    [c, beta, Wi, rsigma2, loglike, loglikepen]=em_MLM_regress(X,Xlogit, Y, cinit,betainit,rsigma2, MLMoption);
  end;
end;

[numcmp2,a2,mu2,sigma2]=GMMFormatConvert(dim,c);
if (strcmp(MLMoption.DISTR,'binomial'))
[pyi,pij]=MLMclassify(a2,mu2,sigma2,beta,X, Xlogit);
%w2 is the intended weights for the two classes to balance the data
w2=[sum(Y==1)/length(Y), sum(Y==0)/length(Y)]; % If no balance is wanted 
%w2=[0.5,0.5]; %If balance of classes is desirec
loglikeY=LoglikehoodYBalance(pyi,Y,w2); %added 8/39/2022
if (MLMoption.verbose>0) 
pyihard=pyi>0.5;
fprintf('Classification accuracy after EM (within training): %f\n', sum(Y'==pyihard)/length(Y));
[xcord,ycord,T,AUC]=perfcurve(Y,pyi',1); % Area under curve
fprintf('Classification AUC after EM (within training): %f\n', AUC);
end;
else
   [pyi,pij]=MLMregress(a2,mu2,sigma2,beta,X, Xlogit); 
   v1=sum((pyi-Y').*(pyi-Y'))/length(Y);
   loglikeY=v1; 
   if (MLMoption.verbose>0) 
     fprintf('Regression MSE after EM (within training): %f\n', v1);
   end;
end;

fprintf('Seed: %d Done---------------------\n', MLMoption.kmseed);



