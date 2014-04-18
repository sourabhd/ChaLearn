function varargout = PCAfeature(varargin)

wsize = varargin{1};
wsize_t = varargin{2};
templateWs = varargin{3};


cnt = 0;
rsize = (wsize-1)/2;
LSK = templateWs(rsize+1:2:size(templateWs,1)-rsize,rsize+1:2:size(templateWs,2)-rsize,1:1:size(templateWs,3),:);
WWW = reshape(LSK,[size(LSK,1)*size(LSK,2)*size(LSK,3),size(LSK,4)]);
Temp = sum(WWW,1);
cnt = size(WWW,1);

meanTemp = repmat(Temp',[1 cnt])./cnt;
size(WWW)

Temp = Temp';
[v,d] = eig((WWW'-meanTemp)*(WWW'-meanTemp)');
[tmp l] = sort(diag(d),'descend');
v = v(:,l);
d = diag(tmp);
for i = 1:size(d,1)
    if trace(d(1:i,1:i))./trace(d) > .8
        valid = i
        break;
    end
end
valid = 4;
% figure(10000),
% for j = 1:wsize_t
%     for i = 1:valid
%         v1(:,:,:,i) = reshape(v(:,i),[wsize,wsize,wsize_t]);
%         temp = v1(:,:,:,i);
%         subplot(valid,wsize_t,(i-1)*wsize_t+j), imagesc(imresize(squeeze(v1(:,:,j,i)),10,'bicubic'),[min(temp(:)),max(temp(:))]),axis off, axis image;
%         %colorbar('SouthOutside'), axis off, axis image;
%     end
% end



TF = zeros(size(templateWs,1),size(templateWs,2),size(templateWs,3),valid);

for i = 1:size(templateWs,1)
    for j = 1:size(templateWs,2)
        for t = 1:size(templateWs,3)
            LSK = templateWs(i,j,t,:);
            TF(i,j,t,:) = ( (LSK(:) - Temp./cnt)'*v(:,1:valid))';
        end
    end
end

varargout{1} = TF;
varargout{2} = Temp./cnt;
varargout{3} = v(:,1:valid);

end
