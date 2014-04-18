function [block,flag,S,E] = Proto_Action(smap,blocksize,thres)

% [RETURNS]
% block : block video (1: salient 0: not salient)
% flag : flag for saliency
% S: starting points of block (top-left)
% E: ending points of block (bottom-right)
%
% [PARAMETERS]
% smap   : saliency map
% blocksize : size of block cube
% thres : threshold for saliency

% [HISTORY]
% May 31, 2011 : created by Hae Jong

[M N T] = size(smap);
% Make integral video of space-time saliency map
smap_integral = cumsum(cumsum(cumsum(smap,1),2),3);
x_block = blocksize(1);
y_block = blocksize(2);
t_block = blocksize(3);

flag = zeros(y_block,x_block,t_block);
block_width = floor(N/x_block);
block_height = floor(M/y_block);
block_time = floor(T/t_block);

% Compute average saliency in the blocks and declare the region if the
% average saliency is over the threshold


block = zeros(M,N,T);
for i = 1:x_block
    for j = 1:y_block
        for t = 1:t_block
            S{i,j,t}.x = (i-1) * block_width + 1;
            S{i,j,t}.y = (j-1) * block_height + 1;
            S{i,j,t}.z = (t-1) * block_time + 1;
            E{i,j,t}.x = i * block_width;
            E{i,j,t}.y = j * block_height;
            E{i,j,t}.z = t * block_time;

            f(i,j,t) = smap_integral(E{i,j,t}.y,E{i,j,t}.x,E{i,j,t}.z) - smap_integral(S{i,j,t}.y,S{i,j,t}.x,S{i,j,t}.z) ...
                - smap_integral(S{i,j,t}.y,E{i,j,t}.x,E{i,j,t}.z) - smap_integral(E{i,j,t}.y,S{i,j,t}.x,E{i,j,t}.z) ...
                - smap_integral(E{i,j,t}.y,E{i,j,t}.x,S{i,j,t}.z) + smap_integral(S{i,j,t}.y,S{i,j,t}.x,E{i,j,t}.z) ...
                + smap_integral(S{i,j,t}.y,E{i,j,t}.x,S{i,j,t}.z) + smap_integral(E{i,j,t}.y,S{i,j,t}.x,S{i,j,t}.z);
            f(i,j,t) = f(i,j,t)/(block_width*block_height*block_time);
            if f(i,j,t) > thres
                if E{i,j,t}.z+t_block <= T
                    block(S{i,j,t}.y:E{i,j,t}.y-1, S{i,j,t}.x:E{i,j,t}.x-1, S{i,j,t}.z+3:E{i,j,t}.z+t_block) = 1;
                else
                    block(S{i,j,t}.y:E{i,j,t}.y-1, S{i,j,t}.x:E{i,j,t}.x-1, S{i,j,t}.z:E{i,j,t}.z) = 1;
                end
                flag(i,j,t) = 1;
            end
        end
    end
end


