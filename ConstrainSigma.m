%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, May 2022                                       %
% For research purpose only.                                      %
%                                                                 %
%-----------------------------------------------------------------%
%                                                                 %
% Function: Restrict the covariance matrix of GMM                 %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%

function sigmaout=ConstrainSigma(a, sigma,dim,numcmp, Xvar, MLMoption);

code=MLMoption.constrain;
ncode=0; % default
if (strcmp(code,'N')) ncode=0; end; % No constraint on the covariance
if (strcmp(code,'EI')) ncode=1; end;% Identical covariance across components, the covariance is scalar matrix (diagonal and identical elements on diagonal)
if (strcmp(code,'VI')) ncode=2; end;% Different covariance across components, the covariance is scalar matrix (diagonal and identical elements on diagonal)
if (strcmp(code,'EEE')) ncode=3; end; % Identical covariance across components, no constraint on the covariance structure
if (strcmp(code,'VVV')) ncode=4; end; % No constraint on the covariance
if (strcmp(code,'DIA')) ncode=5; end; % All covariance matrices are diagonal, but allowed different
if (strcmp(code,'DIAE')) ncode=6; end; % All covariance matrices are diagonal and identical
if (strcmp(code,'DIAS')) ncode=7; end;  % All covariance matrices shrunk towards a diagonal matrix, extent of shrinkage is determined by MLMoption.diagshrink. The covariance are different across components
% Decompose each covariance matrix into orientation, shape, and volumn (size)
if (strcmp(code,'EEV')) ncode=8; end; % Identical shape and volumn across components, different orientation across components 
if (strcmp(code,'VEV')) ncode=9; end;  % Identical shape across components, different volume and orientation across components 

%fprintf('Inside ConstrainSigma, ncode=%d\n',ncode);

if (ncode==0 | ncode==4) % No need to impose constraint
    sigmaout=sigma;
    return;
end;

if (ncode==5)
    for j=1:numcmp
        sigmaout(:,:,j)=diag(diag(sigma(:,:,j))); 
    end;
    return;
end;

if (ncode==7) %shrink towards diagonal
  v1=MLMoption.diagshrink;
  if (v1>1.0) v1=1.0; end;
  if (v1<0.0) v1=0.0; end;
  for j=1:numcmp
    sigmaout(:,:,j)=sigma(:,:,j)*(1-v1)+v1*diag(diag(sigma(:,:,j))); 
  end;
  return;
end;

% Apply regularization depending on the specified constraint

% weighted average covariance matrix is needed
sigmaave=zeros(dim,dim);
for j=1:numcmp
    sigmaave=sigmaave+a(j)*sigma(:,:,j);

end;

if (ncode==6)
    for j=1:numcmp
        sigmaout(:,:,j)=diag(diag(sigmaave(:,:))); 
    end;
    return;
end;

if (ncode==1)
    for j=1:numcmp
        sigmaout(:,:,j)=mean(diag(sigmaave(:,:)))*diag(ones(dim,1));
    end;
    return;
end;

if (ncode==2)
    for j=1:numcmp
        sigmaout(:,:,j)=mean(diag(sigma(:,:,j)))*diag(ones(dim,1));
    end;
    return;
end;

if (ncode==3)
    for j=1:numcmp
        sigmaout(:,:,j)=sigmaave;
    end;
    return;
end;


% Compute the volume, shape, and orientation of each covariance matrix
vscale=zeros(size(a)); % Volume
for j=1:numcmp
    [V(:,:,j), D(:,:,j)]=eig(sigma(:,:,j)); % V is orientation
    vscalelog=0;
    for i=1:dim
        if (D(i,i,j)>0)
        vscalelog=vscalelog+log(D(i,i,j));
        else
            fprintf('Warning:nonpositive eigen value of covariance matrix\n');
            fprintf('Sigma of component %d:\n',j);
            sigma(:,:,j)
            error('ConstrainSigma: Error\n');
        end;
    end;
    vscale(j)=exp(vscalelog/dim);
    D(:,:,j)=D(:,:,j)/vscale(j); % D is now the shape diagonal matrix
end;


j=numcmp+1;
[V(:,:,j), D(:,:,j)]=eig(sigmaave(:,:)); % V is orientation
vscalelog=0;
    for i=1:dim
        if (D(i,i,j)>0)
        vscalelog=vscalelog+log(D(i,i,j));
        else
            fprintf('Warning:nonpositive eigen value of covariance matrix\n');
            fprintf('Weighted average Sigma over the components:\n');
            sigmaave(:,:)
            error('ConstrainSigma: Error\n');
        end;
    end;
vscaleave=exp(vscalelog/dim);
D(:,:,j)=D(:,:,j)/vscaleave; % D is now the shape diagonal matrix


if (ncode==8)
    for j=1:numcmp
        sigmaout(:,:,j)=vscaleave*V(:,:,j)*D(:,:,numcmp+1)*V(:,:,j)';
    end;
end;

if (ncode==9)
    for j=1:numcmp
        sigmaout(:,:,j)=vscale(j)*V(:,:,j)*D(:,:,numcmp+1)*V(:,:,j)';
    end;
end;


for j=1:numcmp
    % Guard against nearly singular matrix
sigmaout(:,:,j)=checksingular(sigmaout(:,:,j), Xvar, 0.05);
end %%% j=1:numcmp

