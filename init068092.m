%%% abc_all[1:92,] # 068/071 (46 each); 93:numdata: 092 (94)
%%% use study indicators as covataites for regression
rawdat=load('Data/MAL.txt','-ascii');
[numdata, dim] = size(rawdat);
Y1=rawdat(:,dim);
X1 = log(rawdat(:, 1:(dim-1)) + 1);
X1s = normalize(X1);
Indi = zeros(numdata, 2); %study indicator
%use 068 as reference
Indi_study(47:92,1) = 1;
Indi_study(93:numdata,2) = 1;
dim = size(X1,2);     %dim without 2 indicators
vargmm = 1:1:dim; % number of features for fitting GMM
ncmp = 2:1:4;
%%% get lambda  value from using the full dataset
rand('state',1);
[B,FitInfo] = lassoglm([X1s,Indi],Y1,'binomial','CV',5, 'Alpha',MLMoption.AlphaLasso);
vlasso =FitInfo.Lambda1SE;
numcmp = 2;
% Specify the GeM-LR model via MLMoption
MLMoption.lambdaLasso=vlasso*ones(1,numcmp);

%==============================================

MLMoption.stopratio=1.0e-5; % Threshold controling the number of EM iterations in em_MLM
MLMoption.kappa=-1.0; % Weights on instances can be part of the optimization if the option is evoked. For the paper, we set negative value that disables the option of weighted instances.
MLMoption.verbose=1; % T
MLMoption.minloop=3; % must be at least 2, otherwise the program automatically uses 2
MLMoption.maxloop=50; % maximum number of iterations in EM
MLMoption.constrain='N';%'N'; %possible strings: 'N' (no constrain), 'EI','VI','EEE','VVV','DIA','DIAE','DIAS','EEV','VEV' from Mclust R package
MLMoption.diagshrink=0.5; %larger value indicates more shrinkage towards diagonal, only used if the constraint is 'DIAS'
MLMoption.kmseed=0;
MLMoption.algorithm=1; % 1 for Lasso, 0 for Logistic without variable selection
%MLMoption.AlphaLasso=0.5; % if set to 1, this is Lasso, value smaller than 1 is elastic net
MLMoption.numcmp=numcmp;
MLMoption.AUC=1; % If set AUC=1, use AUC to pick the best seed in estimateBestSD, otherwise, use accuracy
MLMoption.DISTR='binomial'; % 'binomial' for classification, 'normal' for regression
MLMoption.NOEM=0; % if equal 1, then only run initialization, NO EM update
%so no initial value is assumed.
%If MLMoption.InitCluster is set as a vector of length=data size, it is
%assumed to be the initial cluster labels used by initemMLM. 
MLMoption.InitCluster = zeros(0,0);
MLMoption.Yalpha = 1.0;
