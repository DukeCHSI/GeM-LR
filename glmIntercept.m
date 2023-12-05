%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, September 2022                                 %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
% Function: Fit a linear model with only intercept term. This is  %
% used when the number of data points for estimation is extremely %
% small.                                                          %
%-----------------------------------------------------------------%

function beta=glmIntercept(Y,wt, dim, DISTR);

if (strcmp(DISTR,'binomial'))
 % compute proportion of points in class 1 
  vec=find(Y==1);
  if (sum(wt)>0)
      probclass1=sum(wt(vec))/sum(wt);
  else
      probclass1=0.5;
  end;
  
  if (probclass1>=0.9999)
      probclass1=0.9999;
  else
      if (probclass1<=0.0001)
          probclass1=0.0001;
      end;
  end;
  
  beta=zeros(dim+1,1);
  beta(1)=log(probclass1/(1-probclass1)); % This is the intercept
  return;
end; % if (strcmp(DISTR,'binomial'))

if (strcmp(DISTR,'normal'))
  beta=zeros(dim+1,1);
  if (sum(wt)<=0)
      beta(1)=mean(Y);
  else
      beta(1)=0.0;
      for i=1:length(Y)
          beta(1)=beta(1)+Y(i)*wt(i);
      end;
      beta(1)=beta(1)/sum(wt);
  end;
  return;
end; % if (strcmp(DISTR,'binomial'))

error('In glmIntercept: DISTR improper\n');
