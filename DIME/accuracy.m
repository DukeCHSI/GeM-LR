function [Acc,D] = accuracy(pi,mu,Sigma, index)
% Compute accuracy for the selected dimensions (index) for a specific
% component comp0, 
% in normal mixture model

num = size(index,1); 
[~,k]=size(mu);  D=zeros(k,k, num); % p dimension, k components

pi = pi';
Acc=zeros(k,num);
delta_ncnc = zeros(k,num);

for s = 1:num
    index1 = index(s,:);
    index1(index1 == 0) = [];
    for i=1:k
        for j=i:k
            D(i,j,s)=mvnormpdf(mu(index1,j),mu(index1,i),Sigma(index1,index1,i)+Sigma(index1,index1,j)); 
            D(j,i,s)=D(i,j,s); % D(i,j) = f_ij
        end
    end    
    E = D(:,:,s)-diag(diag(D(:,:,s)));
    %E*pi' is the numerator of delta_c
    dcc = 1./(1-pi).*(E*pi);
    Delta_c =  dcc ./diag(D(:,:,s)); %% delta_c for c = 1:k for normal components
    for i = 1:k
        Dsub =D(:,:,s);
        Dsub(i,:) = [];
        Dsub(:,i) = [];
        pisub = pi;
        pisub(i) = [];

        delta_ncnc(i,s) = (1./(1-pi(i))^2)*sum(sum(Dsub.*(pisub*pisub')));
    end
    Delta_nc =  dcc ./delta_ncnc(:,s); 
    Acc(:,s) = (pi.*pi)./(pi + (1-pi).*Delta_c) + (1-pi).*(1-pi)./(1-pi + pi.*Delta_nc);

end


