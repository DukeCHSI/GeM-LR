%% Initialize the GeMLR model
nseeds = 20; %number of seeds
kkk = 5; % number of folds for CV
MLMoption.AlphaLasso=0.8000; % tuning parameter alpha for the elasticnet
rangeSeed = 30;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% run('initVAST.m') %for reading VAST data and setting hyperparameters for GeM-LR model 
%% run('init068092.m') % for reading MAL (CHMI) data and setting hyperparameters for GeM-LR model 
run('initVAST.m')
%% Get cvAUC 
[cvAUC, ~, ~]=runCV(kkk, ncmp,nseeds,rangeSeed,vargmm, Y1,X1, Indi, MLMoption);
mean(cvAUC)% mean cvAUC for GeM-LR models with different number of components

%% Obtain the final fitted model 
[beta, clusterid, prop, mu, sigma]=finalModel(cvAUC, ncmp,nseeds,rangeSeed,vargmm, Y1,X1s, Indi, MLMoption);
%beta: coefficient for logistic regression
%clusterid: clustering results
%prop, mu,sigma: mixture component prior, mean vectors and covariance
%matrices for GMM model

%% Perform DIME analysis to select discriminative variables for each GMM component
comp = 1; %comp: variable selection for the specified comp (mixture component)
[varsel, wm] = VS(comp, prop ,mu,sigma,length(vargmm));
%varsel: selected variables for mixture component comp
%wm: achieved accuracy


%% for HVTN505
run('init505.m')
% 9: Env gp140–specific IgA ; 7: Env IgA; 5: ADCP; 6: R2
X1 = [X11(:,7),rawdat(:,1:4),X11(:,5)];%ADCP using the cts version of IgA for clustering
X1s = normalize(X1);
cvAUC=runCV_wts(kkk, ncmp,nseeds,rangeSeed,vargmm, Y1,X1,MLMoption, wts,5);
mean(mean(cvAUC),3)% mean cvAUC for GeM-LR models with different number of components

MLMoption.lambdaLasso=MLMoption.lambdaLasso(1)*ones(1,ncmp);
rseeds = datasample(1:1:rangeSeed,nseeds,'Replace',false);
[c, beta, ~, ~, ~, ~, ~,~]=estimateBestSD(X1s(:,vargmm),X1s(:,2:size(X1s,2)),Y1, MLMoption,rseeds); 
[~,a2,mu2,sigma2]=GMMFormatConvert(size(vargmm,2),c);  
[pyi,pij]=MLMclassify(a2,mu2,sigma2,beta,X1s(:,vargmm),X1s(:,2:size(X1s,2)));
[val, clusterid] = max(pij,[],2);

%% Example plot routine to visulize model output
rowNames = {'Intercept','age','BMI','race','bhvrisk', 'ADCP'};
colNames = {'Cluster 1','Cluster 2'};
h = heatmap(beta);
h.YDisplayLabels = rowNames;
h.XDisplayLabels = colNames;
h.CellLabelColor = 'none'; % hides all cell labels
n = 256;
neg = [linspace(0,1,n/2)' ... % R: 0 → 1
linspace(0,1,n/2)' ... % G: 0 → 1
ones(n/2,1)]; % B: 1 → 1 (blue → white)
pos = [ones(n/2,1) ... % R: 1 → 1
linspace(1,0,n/2)' ... % G: 1 → 0
linspace(1,0,n/2)']; % B: 1 → 0 (white → red)
cmap = [neg; pos];
% Apply colormap directly to the heatmap
h.Colormap = cmap;
% Symmetric color limits so 0 is mapped to white
cmax = max(abs(beta(:)));
h.ColorLimits = [-cmax cmax];