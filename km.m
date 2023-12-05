%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, September 2004                                 %
% For research purpose only.                                      %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%

%%--------------------------------------------------------%%%
%% Subroutine for kmeans algorithm.                       %%%
%% dat is the input data with every sample stored in      %%%
%% one row.  nkm is the number of clusters prespecified.  %%%
%% thred is the threshold for determining when to stop    %%%
%% the iteration in kmeans.  Suggested value for thred:   %%%
%% 1.0e-4 to 1.0e-6.                                      %%%
%%--------------------------------------------------------%%%
%-------------------------------------------------------%
% Output:                                               %
% cdbk: stores the cluster centers found by kmeans      %
% ind: stores the cluster indentity of each sample      % 
%-------------------------------------------------------%


function [cdbk, distlist, ind]=km(dat, nkm, thred,kmseed);

[len,dim]=size(dat);

rand('state', kmseed); %7408, 7018
b=rand(len,1);
[b2,ind2]=sort(b);

cdbk=zeros(nkm,dim);

% Randomly choose a few points from the dataset and use them
% to initialize the codebook
for i=1:nkm
  cdbk(i,:)=dat(ind2(i),:);
end;

ind=zeros([len,1]);

aa=zeros(nkm,1);

stdlist=std(dat);
mvlist=mean(dat);
dist=sum(stdlist.*stdlist)*len*10;

done=0;
clear distlist;
while (done~=-1)

newcdbk=zeros(nkm,dim);
newct=zeros(nkm,1);
newdist=0.0;
for i=1:len
for j=1:nkm
aa(j)=sum((dat(i,:)-cdbk(j,:)).^2);
end; %% for j

[minv, ind(i)]=min(aa);
newcdbk(ind(i),:)=newcdbk(ind(i),:)+dat(i,:);
newct(ind(i))=newct(ind(i))+1;
newdist = newdist+minv;

end;  %% for i

for j=1:nkm
    if (newct(j)>0)
newcdbk(j,:)=newcdbk(j,:)/newct(j);
    else % keep the old codeword for empty cluster
        newcdbk(j,:)=cdbk(j,:);
    end;
end;

done=done+1;
distlist(done)=newdist;

if ((dist-newdist)/dist < thred)
done=-1;
end;

dist=newdist;
cdbk=newcdbk;


end; %%% while

ind=ones(len,1);
aa=zeros(nkm,1);

for i=1:len
for j=1:nkm
aa(j)=sum((dat(i,:)-cdbk(j,:)).^2);
end; %% for j

[minv, ind(i)]=min(aa);
end; %% for i

