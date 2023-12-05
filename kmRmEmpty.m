%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%
%                                                                 %
% Creator: Jia Li, September 2004                                 %
% For research purpose only.                                      %
%                                                                 %
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%

%--------------------------------------------------------%%%
% Subroutine for removing empty clusters of kmeans       %%%
% dat is the input data with every sample stored in      %%%
% one row.  nkm is the number of clusters prespecified.  %%%
%--------------------------------------------------------%%%
%-------------------------------------------------------%
% Output:                                               %
% cdbk: stores the cluster centers found by kmeans      %
% ind: stores the cluster indentity of each sample      % 
%-------------------------------------------------------%


function [cdbk, ind]=kmRmEmpty(dat, nkm, cdbkinit);

[len,dim]=size(dat);

if (nkm>len)
    error('Abort: number of clusters=%d is larger than data size %d\n',nkm,len);
end;

ptdist=zeros(1,len);
cdbk=cdbkinit;

for i=1:len
for j=1:nkm
aa(j)=sum((dat(i,:)-cdbk(j,:)).^2);
end; %% for j
[ptdist(i), ind(i)]=min(aa);
end; %% for i

ndatpercls=zeros(1, nkm); 
for i=1:len
    ndatpercls(ind(i))=ndatpercls(ind(i))+1;
end;

numempty=sum(ndatpercls==0);
if (numempty==0)
    return; % No empty cluster, return directly
end;

[ptdist_sort,id2]=sort(ptdist,'descend'); % descending order of ptdist

ii=1;
for i=1:nkm
    if (ndatpercls(i)==0)
        cdbk(i,:)=dat(id2(ii),:); %replace codebook of an empty cluster by a data point with the largest distances
        ii=ii+1;
    end;
end;

% Cluster again by the new codebook
for i=1:len
for j=1:nkm
aa(j)=sum((dat(i,:)-cdbk(j,:)).^2);
end; %% for j
[ptdist(i), ind(i)]=min(aa);
end; %% for i

ndatpercls=zeros(1, nkm); 
for i=1:len
    ndatpercls(ind(i))=ndatpercls(ind(i))+1;
end;

numempty=sum(ndatpercls==0);
if (numempty>0)
    error('Abort: Cannot get rid of empty cluster via kmRmEmpty, data size too small or too many identical data entries\n');
    return; % No empty cluster, return directly
end;


