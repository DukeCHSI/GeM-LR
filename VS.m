function [varsel, wm] = VS(comp, a2,mu2,sigma2,dimgmm)
%comp:  %variable selection for the specified comp (mixture component)
addpath('DIME')  
[index] = creatindex (dimgmm, dimgmm-1);
[Acc,~] = accuracy(a2,mu2,sigma2, index);
Accfull = Acc(:,1);
Acc(:,1) = [];
[wm, vm] = max(Acc(comp,:));
varsel = index(vm+1,1);
remain = 1:dimgmm;
remain(varsel) = [];
Accref = 0.01;

while((wm- Accref)/Accref > 0.1 && abs(wm - Accfull(comp)) > 0.01)
    Accref = wm;
    index = combine(repelem(varsel,length(remain), 1)', remain);
    index = index';
    [Acc,~] = accuracy(a2,mu2,sigma2, index);
    [wm, vm] = max( Acc(comp,:));
    varsel = index(vm,:);
    remain = 1:dimgmm;
    remain(varsel) = [];
end
end
%varsel % selected variables
%wm %achieved accuracy