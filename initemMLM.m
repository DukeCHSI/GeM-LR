%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%

%-----------------------------------------------------------%%
% Subroutine to initianize parameters for the EM algorithm --%
% The input data X contains one sample on each row         --%
%-----------------------------------------------------------%%

function [muinit, sigmainit, ainit,beta,rsigma2]=initemMLM(X, Xlogit, Y, MLMoption);

[numdata,dim]=size(X);
dimlogit=size(Xlogit,2);
numcmp=MLMoption.numcmp(1);

if (numcmp>numdata)
    error('Abort: number of components=%d is larger than data size %d\n',numcmp,numdata);
end;

kmseed=MLMoption.kmseed;
Xvar=var(X);
if (mean(Xvar)<1.0e-6)
    fprintf('Warning: average variance is very small: %f, may lead to singular matrix\n', mean(Xvar));
end;
for j=1:dim
    if (Xvar(j)<1.0e-6)
       fprintf('Warning: variance of dimension %d is very small: %f, may lead to singular matrix\n', j, Xvar(j));
     end;
end;

%Use Kmeans provided by Matlab library
%[cd1, cdbk1]=kmeans(X, numcmp);

if (size(MLMoption.InitCluster,1)==0)
    nloops=10;
else
    nloops=0; % effectively bypass kmeans initialization
end;

for ii=1:nloops
    [cdbk, distlist, cd1]=km(X, numcmp, 1.0e-4,kmseed); 

    ndatpercls=zeros(1, numcmp); 
    for i=1:numdata
        ndatpercls(cd1(i))=ndatpercls(cd1(i))+1;
    end;

    k=sum(ndatpercls==0);
    if (k==0)
        break;
    end;

    fprintf('Warning initemMLM: kmeans generated empty cluster. Rerun kmeans using seed: %d\n',kmseed+ii);

    if (ii==nloops)
        fprintf('Warning initemMLM: kmeans generated empty cluster %d times.\n',nloops);
        fprintf('Warning: failure to generate kmeans with NO empty clusters\n');
        fprintf('Split clusters to generate %d clusters from % data points\n',numcmp,numdata);
        [cdbk, cd1]=kmRmEmpty(X, numcmp, cdbk);
    end;

    kmseed=kmseed+ii;
end;

if (nloops==0)
    cd1=MLMoption.InitCluster(:,1);

    if (max(cd1)>numcmp)
        fprintf('Warning: the number of components %d in MLMoption is smaller than %d appeared in the initial cluster labels\n', numcmp, max(cd1));
        fprintf('Abort initemMLMcls\n');
        return;
    end;
    
    ndatpercls=zeros(1, numcmp); 
    for i=1:numdata
        ndatpercls(cd1(i))=ndatpercls(cd1(i))+1;
    end;

    k=sum(ndatpercls==0);
    if (k>0)
     fprintf('Warning: there are one or more empty clusters in the input initial cluster labels\n');
     fprintf('Abort initemMLMcls\n');
     return; 
    end;
    
end;

rsigma2=zeros(1,numcmp);
ainit=ndatpercls/numdata;
muinit=zeros(dim, numcmp);
for i=1:numdata
muinit(:, cd1(i))=muinit(:,cd1(i))+X(i,:)';
end;

for j=1:numcmp
muinit(:,j)=muinit(:,j)/ndatpercls(j);
end;

sigmainit=zeros(dim,dim,numcmp);
for i=1:numdata
sigmainit(:,:,cd1(i))=sigmainit(:,:,cd1(i))+(X(i,:)'-muinit(:,cd1(i)))*(X(i,:)'-muinit(:,cd1(i)))';
end;

for j=1:numcmp
if (ndatpercls(j)>0)
    sigmainit(:,:,j)=sigmainit(:,:,j)/ndatpercls(j);
end;


sigmainit(:,:,j)=checksingular(sigmainit(:,:,j), Xvar, 0.05);

end;

for j=1:numcmp
numdata_small=sum(cd1==j);
Ysmall=Y(find(cd1==j));
Xsmall=Xlogit(find(cd1==j),:);
if (numdata_small>=3)
[beta_nointercept(:,j),STATS]=lassoglm(Xsmall,Ysmall,MLMoption.DISTR,'Lambda',[MLMoption.lambdaLasso(j)],'Alpha',MLMoption.AlphaLasso);
beta(:,j)=[STATS.Intercept; beta_nointercept(:,j)];
else
  beta(:,j)=glmIntercept(Ysmall,ones(length(Ysmall),1), dimlogit, MLMoption.DISTR); %use strcmp to compare strings
end;

if (strcmp(MLMoption.DISTR,'normal')) % case of regression
  rsigma2(j)=0;
  for i=1:numdata_small
    v1=Xsmall(i,:)*beta(2:dimlogit+1,j)+beta(1,j);
    rsigma2(j)=rsigma2(j)+(Ysmall(i)-v1)*(Ysmall(i)-v1);
  end;
end;

end;

% Identical noise variance for all the components
v1=sum(rsigma2);
rsigma2=ones(size(rsigma2))*v1/numdata;

