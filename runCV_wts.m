function [cvAUCfinal]=runCV_wts(kkk, ncmp,nseeds,rangeSeed,vargmm, Y1,X1, MLMoption, wts, iter)
dimgmm = length(vargmm);
lcmp = length(ncmp);
cvAUCfinal = zeros(kkk,lcmp, iter);
dim = size(X1,2);
for it  = 1:iter
    rand('state', (8+it));
    tuningK = cvpartition(Y1,'KFold',kkk); 
    rseeds = datasample(1:1:rangeSeed,nseeds,'Replace',false);
    for ifold = 1:kkk
        Xtraining_raw = ipOversampling([X1(tuningK.training(ifold),:), Y1(tuningK.training(ifold))],wts(tuningK.training(ifold)), false);
        [Xtraining,Ctest1,Stest1]  = normalize(Xtraining_raw(:,1:dim));
        Ytt = Y1(tuningK.test(ifold));
        Ytraining = Xtraining_raw(:,dim+1);
        Xtt = (X1(tuningK.test(ifold),:) - Ctest1)./Stest1; % testing
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for  jj = 1:lcmp
            MLMoption.numcmp = ncmp(jj);
            MLMoption.lambdaLasso=MLMoption.lambdaLasso(1)*ones(1,ncmp(jj));
            [c, beta, ~, ~, ~, ~, ~,~]=estimateBestSD(Xtraining(:,vargmm),Xtraining(:,2:dim),Ytraining, MLMoption,rseeds); % revised 8/30/2022
            [~,a2,mu2,sigma2]=GMMFormatConvert(dimgmm,c);  

            [pyi,~]=MLMclassify(a2,mu2,sigma2,beta,Xtt(:,vargmm),Xtt(:,2:dim));

            [~,~,~,cvAUCfinal(ifold,jj, it), tm]=perfcurve(Ytt,pyi',1, 'Weights',wts(tuningK.test(ifold))); % weighted AUC

         end 
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    %%%%%
   end

end
end