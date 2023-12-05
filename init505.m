rawdat=load('Data/hvtn505_nolog_race_wts.txt','-ascii'); %1:4 covariates age, BMI, race and bhvrisk; last Y; dim-1 weights
[numdata, dim] = size(rawdat);
Y1=rawdat(:,dim);
X11 = rawdat(:,5:(dim-2));
wts = rawdat(:,dim-1);
% 9: Env gp140–specific IgA ; 7: Env IgA; 5: ADCP; 6: R2
%dim without the indicator
kkk = 5; % number of folds for CV
iter = 5;
vargmm = 1;

ncmp = 2; % number of mixture components
numcmp = 2;

vlasso = 0;
MLMoption.Yalpha = 1.0;

MLMoption.stopratio=1.0e-5; % Threshold controling the number of EM iterations in em_MLM
% If MLMoption.kappa is negative or zero, the program will not evoke instance
% weighted estimation. Equivalently, if kappa is very large, the instance
% tends to be equally weighted, equivalent to not weighted version
MLMoption.kappa=-1.0; % This must be experimented with, larger kappa yields closer to uniform weights, negative value disables the option of weighted instances
MLMoption.verbose=0; % T
MLMoption.minloop=3; % must be at least 2, otherwise the program automatically uses 2
MLMoption.maxloop=50; % maximum number of iterations in EM
MLMoption.constrain='N';%'N'; %possible strings: 'N' (no constrain), 'EI','VI','EEE','VVV','DIA','DIAE','DIAS','EEV','VEV'
MLMoption.diagshrink=0.5; %larger value indicates more shrinkage towards diagonal, only used if the constraint is 'DIAS'
MLMoption.kmseed=0;
MLMoption.algorithm=1; % 1 for Lasso, 0 for Logistic without variable selection
MLMoption.numcmp=numcmp;
MLMoption.AlphaLasso=1;
MLMoption.lambdaLasso=vlasso*ones(1,numcmp); % different values can be tried
MLMoption.AUC=1; % If set AUC=1, use AUC to pick the best seed in estimateBestSD, otherwise, use accuracy
MLMoption.DISTR='binomial'; % 'binomial' for classification, 'normal' for regression
MLMoption.NOEM=0; % if equal 1, then only run initialization, NO EM update
%so no initial value is assumed.
%If MLMoption.InitCluster is set as a vector of length=data size, it is
%assumed to be the initial cluster labels used by initemMLM. 
MLMoption.InitCluster = zeros(0,0);
