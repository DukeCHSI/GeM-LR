%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
% Function: Compute the conditional log likelihood of Y given X.  %
% As the classes may be very unbalanced, adjust the likelihood    %
% by assuming the class proportions are given in input w. If w is %
% set to the empirical proportions of the classes, the output     %
% is simply the conditional log likelihood of the observed Y.     %
%-----------------------------------------------------------------%

function loglikeY=LoglikehoodYBalance(pyi,Y,w)

numdata=length(pyi);

loglike=zeros(2,1);
for i=1:numdata
if (Y(i)==1)
    if (pyi(i)>0.0001)
    loglike(1)=loglike(1)+log(pyi(i));
    else % artificially treat extreme values
        loglike(1)=loglike(1)+log(0.0001);
    end;
else % Y(i)==0)
    if (1-pyi(i)>0.0001)
    loglike(2)=loglike(2)+log(1-pyi(i));
    else % artificially treat extreme values
        loglike(2)=loglike(2)+log(0.0001);
    end;
end;

end %% i=1:numdata

n1=sum(Y==1);
n=length(Y);

if (sum(w)<=0)
    fprintf('LoglikehoodYBalance: Input weight w has non-positive sum, [%f, %f]',w(1),w(2));
    error('Exit LoglikehoodYBalance\n');
end;

w=w/sum(w); % normalization
if (n1>0 & n-n1>0)
loglikeY=n*(loglike(1)/n1*w(1)+loglike(2)/(n-n1)*w(2));
else
    loglikeY=sum(loglike);
    fprintf('Warning LoglikehoodYBalance: Input Y has only one class label, loglikelhood of Y not balanced\n');
end;
