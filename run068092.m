%%%% multiple kmean seeds for the malaria data
%%%% no prior model: alpha = 0
%%%% use the full data to find a single lambda
 %%% running different data using the same working file: seeds_cv_work.m 08/31/2022
 %%% Using full features for GMM and a set of number of components
 %%% using training error to select seed for each fold
%%% abc_all[1:92,] # 068/071 (46 each); 93:numdata: 092 (94)
%%% use study indicators as covataites for regression

rawdat=load('Data/MAL.txt','-ascii');
[numdata, dim] = size(rawdat);
Y1=rawdat(:,dim);
X1 = log(rawdat(:, 1:(dim-1)) + 1);
X1s = normalize(X1);
Indi_study = zeros(numdata, 2); %study indicator
%use 068 as reference
Indi_study(47:92,1) = 1;
Indi_study(93:numdata,2) = 1;
dim = size(X1,2);     %dim without 2 indicators
nseeds = 20;
kkk = 5; % number of folds for CV
vargmm = 1:1:dim; % number of features for fitting GMM
dimgmm = length(vargmm);

MLMoption.AlphaLasso=0.8000;
MLMoption.Yalpha = 1.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ncmp = 2:1:4;
lcmp = length(ncmp);
cvAUCfinal = zeros(kkk, lcmp);%record the cvAUC for each fold and each model with number of components = ncmp
bestseed= zeros(kkk, lcmp);
numcmp = 2;
run("init068092.m")
MLMoption.Yalpha = 1.0;
rangeSeed = 30;
labels = cell(lcmp, kkk);
guess = cell(lcmp, kkk);

rand('state', 9);
tuningK = cvpartition(Y1,'KFold',kkk); 
rseeds = datasample(1:1:rangeSeed,nseeds,'Replace',false);   
for ifold = 1:kkk
    [Xtraining,Ctest1,Stest1]  = normalize(X1(tuningK.training(ifold),:));
    Ytt = Y1(tuningK.test(ifold));
    Ytraining = Y1(tuningK.training(ifold));
    Xtt = (X1(tuningK.test(ifold),:) - Ctest1)./Stest1; % testing
    Xtrain_indi = [Xtraining Indi_study(tuningK.training(ifold),:)];
    Xtt_indi = [Xtt Indi_study(tuningK.test(ifold),:)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for  jj = 1:lcmp
        MLMoption.numcmp = ncmp(jj);
        MLMoption.lambdaLasso=MLMoption.lambdaLasso(1)*ones(1,ncmp(jj));
        [c, beta, ~, ~, ~, ~, ~,bestseed(ifold, jj)]=estimateBestSD(Xtraining(:,vargmm),Xtrain_indi,Ytraining, MLMoption,rseeds); % revised 8/30/2022
        [~,a2,mu2,sigma2]=GMMFormatConvert(dimgmm,c);  
        [pyi,pij]=MLMclassify(a2,mu2,sigma2,beta,Xtt(:,vargmm),Xtt_indi);
             
        labels{jj, ifold} = Ytt;
        guess{jj, ifold} = pyi;

        [~,~,~,cvAUCfinal(ifold,jj)]=perfcurve(Ytt,pyi',1);
    end
end

mean(cvAUCfinal)% mean cvAUC for GeM-LR models with different number of components

%%% obtain the final fitted model

[maxvf, maxcmp] = max(mean(cvAUCfinal));
MLMoption.numcmp = ncmp(maxcmp);
MLMoption.lambdaLasso=MLMoption.lambdaLasso(1)*ones(1,ncmp(maxcmp));
rand('state', 9);
rseeds = datasample(1:1:rangeSeed,nseeds,'Replace',false);
[c, beta, ~, ~, ~, ~, ~,~]=estimateBestSD(X1s(:,vargmm),[X1s Indi_study],Y1, MLMoption,rseeds); % beta: cluster-specific regression coefficients
[~,a2,mu2,sigma2]=GMMFormatConvert(dimgmm,c);   % parameters associated with the GMM
[pyi,pij]=MLMclassify(a2,mu2,sigma2,beta,X1s(:,vargmm),[X1s Indi_study]);
[~,~,~,cvAUCfull]=perfcurve(Y1,pyi',1);
[val, clusterid] = max(pij,[],2); %clusterid: the clustering results



addpath('DIME')  
comp =1;
[index] = creatindex (dim, dim-1);
[Acc,D] = accuracy(a2,mu2,sigma2, index);
Accfull = Acc(:,1);
Acc(:,1) = [];
[wm, vm] = max(Acc(comp,:));
varsel = index(vm+1,1);
remain = 1:dim;
remain(varsel) = [];
Accref = 0.01;

while((wm- Accref)/Accref > 0.1 && abs(wm - Accfull(comp)) >= 0.01)
    Accref = wm;
    index = combine(repelem(varsel,length(remain), 1)', remain);
    index = index';
    [Acc,D] = accuracy(a2,mu2,sigma2, index);
    [wm, vm] = max( Acc(comp,:));
    varsel = index(vm,:);
    remain = 1:dim;
    remain(varsel) = [];
end

varsel % selected variables
wm %achieved accuracy
