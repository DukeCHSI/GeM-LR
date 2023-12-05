function [cvAUCfinal, labels, guess]=runCV(kkk, ncmp,nseeds,rangeSeed,vargmm, Y1,X1, Indi, MLMoption)
 %% Run CV for GeM-LR with number of components being ncmp
 lcmp = length(ncmp); %number of GeM-LR models 
 dimgmm = length(vargmm);
 labels = cell(lcmp, kkk);
 guess = cell(lcmp, kkk);
 cvAUCfinal = zeros(kkk, lcmp);%record the cvAUC for each fold and each model with number of components = ncmp
 bestseed= zeros(kkk, lcmp);
 rand('state', 9);
 tuningK = cvpartition(Y1,'KFold',kkk); %cv partition
 rseeds = datasample(1:1:rangeSeed,nseeds,'Replace',false);
 dim =size(X1,2);
 for ifold = 1:kkk

    [Xtraining,Ctest1,Stest1]  = normalize(X1(tuningK.training(ifold),:));
    Ytraining = Y1(tuningK.training(ifold));
    
    Ytt = Y1(tuningK.test(ifold));
    Xtt = (X1(tuningK.test(ifold),:) - Ctest1)./Stest1; % testing
    if(isempty(Indi))
        Xtrain_indi = Xtraining;
        Xtt_indi = Xtt;
    else
        Xtrain_indi = [Xtraining Indi(tuningK.training(ifold),:)];
        Xtt_indi = [Xtt Indi(tuningK.test(ifold),:)];
    end

        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for  jj = 1:lcmp
        MLMoption.numcmp = ncmp(jj);
        MLMoption.lambdaLasso=MLMoption.lambdaLasso(1)*ones(1,ncmp(jj));
        [c, beta, ~, ~, ~, ~, ~,bestseed(ifold, jj)]=estimateBestSD(Xtraining(:,vargmm),Xtrain_indi,Ytraining, MLMoption,rseeds); % revised 8/30/2022
        MLMoption.kmseed = bestseed(ifold,jj);
        [~,a2,mu2,sigma2]=GMMFormatConvert(dimgmm,c);  
  
         [pyi,pij]=MLMclassify(a2,mu2,sigma2,beta,Xtt(:,vargmm),Xtt_indi);

         labels{jj, ifold} = Ytt;
         guess{jj, ifold} = pyi;

         [~,~,~,cvAUCfinal(ifold,jj)]=perfcurve(Ytt,pyi',1);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end
end

    

