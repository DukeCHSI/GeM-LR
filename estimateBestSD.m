%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
%                                                                 %
% Function: Repeat EM algorithm for estimating MLM using multiple %
% random seeds for kmeans initialization. Choose the best seed    %
% based on training accuracy or AUC.                              %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
 
function [c, beta, rsigma2, loglike, loglikepen, loglikeY, Wi,bestseed]=estimateBestSD(X,Xlogit,Y, MLMoption,seedlist); % revised 8/30/2022

  if (sum(seedlist<0)>0)
    fprintf('Warning: seedlist contains negative value, NOT recommended, can confuse the meaning of output bestseed\n');
  end;
  
nseedkm=length(seedlist);
[numdata,dim]=size(X);
dimlogit=size(Xlogit,2);
InitCluster=MLMoption.InitCluster;
[m,numinit]=size(InitCluster); % every column is one initial clustering result, #columns is #initializations
nseed=nseedkm+numinit;

if (nseed<1)
  fprintf('Error: number of initializations is %d\n',nseed);
  fprintf('Input correct seedlist or MLMoption.InitCluster\n');
  return;
end;

accuracy=zeros(nseed,1);
AUC=zeros(nseed,1);

for ii=1:nseed
  if (ii<=nseedkm) % run kmeans
    MLMoption.InitCluster=zeros(0,0); % turn off so that kmeans with a seed will run
    MLMoption.kmseed=seedlist(ii);
  else
    MLMoption.InitCluster=InitCluster(:,ii-nseedkm);
    MLMoption.kmseed=0;
  end;
  
  [c, beta, rsigma2, loglike, loglikepen, loglikeY, Wi]=estimate(X,Xlogit,Y, MLMoption);
  [numcmp2,a2,mu2,sigma2]=GMMFormatConvert(dim,c);
       
  if (strcmp(MLMoption.DISTR,'binomial')) % Case of classification
    [pyi,pij]=MLMclassify(a2,mu2,sigma2,beta,X, Xlogit);
    pyihard=pyi>0.5; %Hard classification
    accuracy(ii)=sum(Y'==pyihard)/length(Y); % percentage of correct classification
    [xcord,ycord,T,AUC(ii)]=perfcurve(Y,pyi',1); % Area under curve
  else % Case of regression
    [pyi,pij]=MLMregress(a2,mu2,sigma2,beta,X, Xlogit); 
    accuracy(ii)=sum((pyi-Y').*(pyi-Y'))/length(Y); %MSE for regression
  end;

end;

if (nseed==1)
  if (nseedkm>0)
    bestseed=seedlist(1);
  else
    bestseed=-1;% negative value indicates the best is not from kmeans seed
  end;
  return;
end;

% Pick the best seed used in kmeans, then refit the model
if (MLMoption.AUC==1 & strcmp(MLMoption.DISTR,'binomial')) % Use AUC to pick
     [v,bestID]=max(AUC);
else % use accuracy to pick
    if (strcmp(MLMoption.DISTR,'binomial')) % Case of classification
        [v,bestID]=max(accuracy);
    else % Case of regression
        [v,bestID]=min(accuracy);
    end;
end;

				% Refit using the best seed
if (bestID<=nseedkm)
  bestseed=seedlist(bestID);
  MLMoption.kmseed=bestseed;
  MLMoption.InitCluster=zeros(0,0);
  [c, beta, rsigma2, loglike, loglikepen, loglikeY, Wi]=estimate(X,Xlogit,Y, MLMoption);
else
  bestseed=bestID-nseedkm;
  MLMoption.InitCluster=InitCluster(:,bestseed);
  MLMoption.kmseed=0;
  [c, beta, rsigma2, loglike, loglikepen, loglikeY, Wi]=estimate(X,Xlogit,Y, MLMoption);
  bestseed=-bestseed; % negative value indicates the best is not from kmeans seed
end;
    
MLMoption.InitCluster=InitCluster; % recover what was originally in MLMoption.InitCluster

