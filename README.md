# GeM-LR: A tool for discovering predictive biomarkers for small datasets in vaccine studies

## Overview:

This repository contains the Matlab codes for estimating GeM-LR by EM algorithm and perform discriminative variable selection (DIME). This tool is the results of the publication entitled "GeM-LR: Discovering predictive biomarkers for small datasets in vaccine studies". 

When using this code in your research, please cite the publication: Lin L, Spreng RL, Seaton KE, Dennison SM, Dahora LC, Schuster DJ, Sawant S, Gilbert P, Fong Y, Kisalu N, Pollard AJ, Tomaras GD, and Li J. GeM-LR: Discovering predictive biomarkers for small datasets in vaccine studies. JOURNAL NAME, VOLUME, PAGE. https://doi.org/XX.XXXX/...

Alternatively, please cite this repository as Lin L, Spreng RL and Li J. GeM-LR: An EM based algorithm for performing discriminative variable selection (DIME)[Computer Software](2023).

## Getting Started:

### Dependencies
This package uses example data files, which are located in the directory ../Data. If users want to use their own data file, it is recommended to be put in the same directory ../Data.

### Instructions
1. Create a directory called "code", unpack the downloaded files and put them under the directory "code". In the discussions below, we assume "code" is the current directory. 

2. In Matlab console, run "master.m" located under the "code" directory. The file outputs cvAUC for 5-fold cross validation under models with different numbers of mixture components (using runCV or runCV_wts functions), and the final model is chosen according to the best cvAUC (the finalModel model). It also outputs the discriminative variable selection for a given mixture component (using VS function). The codes associated with the DIME algorithm are located in ../DIME. 
 
3. Remarks on usage:

  3.1. If MLMoption.DISTR=='binomial', classification is performed. If MLMoption.DISTR=='normal', regression is performed.

  3.2. Performance of classification can be evaluated by AUC or classification accuracy.

  3.3. Performance of regression is evaluated by mean squared error, that is, the sum of squared residual normalized by the number of data points.

  3.4. MLMoption.NOEM=1 (default is 0). If set to 1, EM update of the mixture model is bypassed, the model formed by an initialization scheme is used. Specifically, k-means clustering is applied to partition the data, and then a Gaussian distribution is estimated for each cluster. The mixture model takes these Gaussian components and use the cluster proportions as the prior probabilities for the components.
