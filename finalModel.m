function [beta, clusterid, a2, mu2, sigma2]=finalModel(cvAUCfinal, ncmp,nseeds,rangeSeed,vargmm, Y1,X1s, Indi, MLMoption)
 dimgmm = length(vargmm);

%%% obtain the final fitted model
[maxvf, maxcmp] = max(mean(cvAUCfinal));
MLMoption.numcmp = ncmp(maxcmp);
MLMoption.lambdaLasso=MLMoption.lambdaLasso(1)*ones(1,ncmp(maxcmp));
rand('state', 9);
rseeds = datasample(1:1:rangeSeed,nseeds,'Replace',false);
[c, beta, ~, ~, ~, ~, ~,~]=estimateBestSD(X1s(:,vargmm),[X1s Indi],Y1, MLMoption,rseeds); % beta: cluster-specific regression coefficients
[~,a2,mu2,sigma2]=GMMFormatConvert(dimgmm,c);   % parameters associated with the GMM
[~,pij]=MLMclassify(a2,mu2,sigma2,beta,X1s(:,vargmm),[X1s Indi]);
[~, clusterid] = max(pij,[],2); %clusterid: the clustering results

end 