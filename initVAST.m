rawdat = load('Data/VASTd0_Indi.txt'); %1st column is the dummy variable for vaccine: Vi-PS = 1
[numdata, dim] = size(rawdat);
Y1=rawdat(:,dim);
X1 = rawdat(:,2:(dim-1));
Indi = rawdat(:,1); % dummy variable for vaccine
dim = size(X1,2);
X1s =  normalize(X1);
[~,vargmm] = maxk(var(X1),5);% the top 5 highly variable features for GMM
ncmp = 2:1:4;
%%% get lambda  value from using the full dataset
rand('state',5);
[B,FitInfo] = lassoglm([X1s,Indi],Y1,'binomial','CV',5, 'Alpha',MLMoption.AlphaLasso);
vlasso = FitInfo.LambdaMinDeviance; 
numcmp = 2;
% Specify the GeM-LR model via MLMoption

MLMoption.lambdaLasso=vlasso*ones(1,numcmp);
%==============================================

MLMoption.stopratio=1.0e-5; % Threshold controling the number of EM iterations in em_MLM
MLMoption.kappa=-1.0; % Weights on instances can be part of the optimization if the option is evoked. For the paper, we set negative value that disables the option of weighted instances.
MLMoption.verbose=1; % T
MLMoption.minloop=3; % must be at least 2, otherwise the program automatically uses 2
MLMoption.maxloop=50; % maximum number of iterations in EM
MLMoption.constrain='DIAS';%'N'; %possible strings: 'N' (no constrain), 'EI','VI','EEE','VVV','DIA','DIAE','DIAS','EEV','VEV' from Mclust R package
MLMoption.diagshrink=0.9; %larger value indicates more shrinkage towards diagonal, only used if the constraint is 'DIAS'
MLMoption.kmseed=0;
MLMoption.algorithm=1; % 1 for Lasso, 0 for Logistic without variable selection
%MLMoption.AlphaLasso=0.5; % if set to 1, this is Lasso, value smaller than 1 is elastic net
MLMoption.numcmp=numcmp;
%MLMoption.lambdaLasso=lambdaLassolist(1)*ones(1,numcmp); % different values can be tried
MLMoption.AUC=1; % If set AUC=1, use AUC to pick the best seed in estimateBestSD, otherwise, use accuracy
MLMoption.DISTR='binomial'; % 'binomial' for classification, 'normal' for regression
MLMoption.NOEM=0; % if equal 1, then only run initialization, NO EM update
%so no initial value is assumed.
%If MLMoption.InitCluster is set as a vector of length=data size, it is
%assumed to be the initial cluster labels used by initemMLM. 
MLMoption.InitCluster = zeros(0,0);
MLMoption.Yalpha = 1.0;