function LARK = ThreeDLARK1(varargin)

% Compute 3-D LARK descriptors

% [RETURNS]
% LARK   : 3-D LARK descriptors
%
% [PARAMETERS]
% input   : Input sequence
% wsize : LARK window size
% alpha : Sensitivity parameter
% h     : smoothing paramter

% [HISTORY]
% Apr 25, 2011 : created by Hae Jong

input = varargin{1};
param = varargin{2};

wsize = param.wsize;
h = param.h;
wsize_t = param.wsize_t;
alpha = param.alpha;
interval = param.interval;

% Gradient calculation
[zx,zy,zt] = gradient(input);

[M,N,T] = size(input);
win = (wsize-1)/2;
win_t = (wsize_t-1)/2;
zx = EdgeMirror3(zx,[win,win,win_t]);
zy = EdgeMirror3(zy,[win,win,win_t]);
zt = EdgeMirror3(zt,[win,win,win_t]);


[x1,x2,x3] = meshgrid(-win:win,-win:win,-win_t:win_t);

for k = 1:wsize_t
    K(:,:,k) = fspecial('disk',win);
end

K = permute(K,[1 3 2]);

for k = 1:wsize
    K(:,:,k) = K(:,:,k).*fspecial('gaussian',[wsize wsize_t],1);
end
K = permute(K,[3 2 1]);
for k = 1:wsize
    K(:,:,k) = K(:,:,k).*fspecial('gaussian',[wsize wsize_t],1);
end
K = permute(K,[1 3 2]);

for k = 1:wsize_t
K(:,:,k) = K(:,:,k)./K(win+1,win+1,k);
end
len = sum(K(:));
lambda = 1;
% Covariance matrices computation
for i = 1 : interval: M
    for j = 1 :interval:  N
        for k = 1 : T
            gx = zx(i:i+wsize-1, j:j+wsize-1, k:k+wsize_t-1).*K;
            gy = zy(i:i+wsize-1, j:j+wsize-1, k:k+wsize_t-1).*K;
            gt = zt(i:i+wsize-1, j:j+wsize-1, k:k+wsize_t-1).*K;
            G = [gx(:), gy(:), gt(:)];
            
            [u s v] = svd(G,'econ');
            S(1) = (s(1,1) + lambda) / ( sqrt(s(2,2)*s(3,3)) + lambda);
            S(2) = (s(2,2) + lambda) / ( sqrt(s(1,1)*s(3,3)) + lambda);
            S(3) = (s(3,3) + lambda) / ( sqrt(s(2,2)*s(1,1)) + lambda);
            tmp = (S(1) * v(:,1) * v(:,1).' + S(2) * v(:,2) * v(:,2).' + S(3) * v(:,3) * v(:,3).')  * ((s(1,1) * s(2,2)*s(3,3) + 0.0000001) / len)^alpha;
            C11(i,j,k) = tmp(1,1);
            C12(i,j,k) = tmp(1,2);
            C22(i,j,k) = tmp(2,2);
            C13(i,j,k) = tmp(1,3);
            C23(i,j,k) = tmp(2,3);
            C33(i,j,k) = tmp(3,3);
            
        end
    end
end
clear zx zy zt;
C11 = C11(1:interval:end,1:interval:end,:);
C12 = C12(1:interval:end,1:interval:end,:);
C22 = C22(1:interval:end,1:interval:end,:);
C23 = C23(1:interval:end,1:interval:end,:);
C33 = C33(1:interval:end,1:interval:end,:);
C13 = C13(1:interval:end,1:interval:end,:);

[M,N,T] = size(C11);
C11 = EdgeMirror3(C11,[win,win,win_t]);
C12 = EdgeMirror3(C12,[win,win,win_t]);
C22 = EdgeMirror3(C22,[win,win,win_t]);
C23 = EdgeMirror3(C23,[win,win,win_t]);
C33 = EdgeMirror3(C33,[win,win,win_t]);
C13 = EdgeMirror3(C13,[win,win,win_t]);


for n = 1 : M
    for m = 1 : N
        for k = 1 : T
            tt = x1 .* ( C11(n:n+ wsize -1, m:m+ wsize -1, k:k+ wsize_t -1) .* x1...
                + C12(n:n+wsize-1, m:m+wsize-1, k:k+wsize_t-1) .* x2 ...
                + C13(n:n+wsize-1, m:m+wsize-1, k:k+wsize_t-1) .* x3 )...
                + x2 .* ( C12(n:n+wsize-1, m:m+wsize-1, k:k+wsize_t-1) .* x1...
                + C22(n:n+wsize-1, m:m+wsize-1, k:k+wsize_t-1) .* x2 ...
                + C23(n:n+wsize-1, m:m+wsize-1, k:k+wsize_t-1) .* x3 )...
                + x3 .* ( C13(n:n+wsize-1, m:m+wsize-1, k:k+wsize_t-1) .* x1...
                + C23(n:n+wsize-1, m:m+wsize-1, k:k+wsize_t-1) .* x2 ...
                + C33(n:n+wsize-1, m:m+wsize-1, k:k+wsize_t-1) .* x3 );
            W = exp(-(0.5/h^2) * tt);
            
                    LARK(n,m,k,:) = W(:)/sum(W(:));
               
        end
    end
end


end