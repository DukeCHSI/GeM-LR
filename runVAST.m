



mean(cvAUCfinal)% mean cvAUC for GeM-LR models with different number of components


%%% obtain the final fitted model
[maxvf, maxcmp] = max(mean(cvAUCfinal));
MLMoption.numcmp = ncmp(maxcmp);
MLMoption.lambdaLasso=MLMoption.lambdaLasso(1)*ones(1,ncmp(maxcmp));
rand('state', 9);
rseeds = datasample(1:1:rangeSeed,nseeds,'Replace',false);
[c, beta, ~, ~, ~, ~, ~,~]=estimateBestSD(X1s(:,vargmm),[X1s Indi_vaccine],Y1, MLMoption,rseeds); % beta: cluster-specific regression coefficients
[~,a2,mu2,sigma2]=GMMFormatConvert(dimgmm,c);   % parameters associated with the GMM
[pyi,pij]=MLMclassify(a2,mu2,sigma2,beta,X1s(:,vargmm),[X1s Indi_vaccine]);
[~,~,~,cvAUCfull]=perfcurve(Y1,pyi',1);
[val, clusterid] = max(pij,[],2); %clusterid: the clustering results



addpath('DIME')  
comp =1; %variable selection for the specified comp (mixture component)
[index] = creatindex (dimgmm, dimgmm-1);
[Acc,D] = accuracy(a2,mu2,sigma2, index);
Accfull = Acc(:,1);
Acc(:,1) = [];
[wm, vm] = max(Acc(comp,:));
varsel = index(vm+1,1);
remain = 1:dimgmm;
remain(varsel) = [];
Accref = 0.01;

while((wm- Accref)/Accref > 0.1 && abs(wm - Accfull(comp)) > 0.01)
    Accref = wm;
    index = combine(repelem(varsel,length(remain), 1)', remain);
    index = index';
    [Acc,D] = accuracy(a2,mu2,sigma2, index);
    [wm, vm] = max( Acc(comp,:));
    varsel = index(vm,:);
    remain = 1:dimgmm;
    remain(varsel) = [];
end


varsel % selected variables
wm %achieved accuracy



