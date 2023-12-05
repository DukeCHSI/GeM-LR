function [index] = creatindex ( dimofx, m)
% creat index of subsets of markers
% dimofx = number of markers
% m = number of markers to be deleted
% m == 0 gives all possible subsets of markers

index = [];
if m == 0
  for mm = 1:dimofx-1  % number of markers to be deleted
     index1 = nchoosek(1:dimofx, (dimofx-mm));  % each row gives a subset of markers
     index = combine(index, index1); % each row gives a subset of markers
  end

else
 index= nchoosek(1:dimofx, (dimofx-m));  
end
index = combine(1:dimofx,index);
