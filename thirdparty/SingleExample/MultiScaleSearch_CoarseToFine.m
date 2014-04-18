function [RMs, Ts,RM2s,RM3s,SC_t] = MultiScaleSearch_CoarseToFine(Q,T,F_Q,F_T,S,E,flag)

% [PARAMETERS]
% Q : query
% T : target
% F_Q : query_feature
% F_T : target_feature
% S, E : starting and ending points of blocks
% flag : saliency block flag
% alpha : confidence level for significance level
% blocksize: block size

tic;

M_Q = size(Q,1); N_Q = size(Q,2);
M_T = size(T,1); N_T = size(T,2);

stepsize = min(N_Q,M_Q); % you can have finer stepsize by having below 0.6
for kkk = 1:20
    if max(M_T,N_T) < 400
        SC_t(kkk) = (400+stepsize*(kkk-15)*1.5)/400;
    elseif max(M_T,N_T) > 600
        SC_t(kkk) = (600+stepsize*(kkk-15)*1.5)/600;
    else
        SC_t(kkk) = (max(M_T,N_T)+stepsize*(kkk-15)*1.5)/max(M_T,N_T);
    end
end
SC_t = SC_t(SC_t>0.8); %0.1
SC_t = SC_t(SC_t<1.6); %1.6

space = 1;  % sampling factor for feature matrix (space)
space_t = 1;  % sampling factor for feature matrix (time)
interval = 1;  % search at 1 pixel apart
SC = [0.2 0.4]; % coarse-to-fine search scale factor

fprintf('Multi-scale search.\n');
progress = 0;

for n = 1:length(SC_t)
    n
    flags = flag;
    for m = 1:length(SC)
        % Rescale query and query feature
        Qs = imresize(Q,SC(m));
        F_Qs = imresize(F_Q,SC(m));
        
        F_Q1 = F_Qs(1:space:end, 1:space:end,1:space_t:end,:);
        F_Q2 = F_Q1(:,end:-1:1,:,:);
        norm_FQ = norm(F_Q1(:),'fro');
        F_Ts = imresize(F_T,SC_t(n)*SC(m),'lanczos3');
        Ts1 = imresize(T,SC_t(n)*SC(m),'lanczos3');
        Ts{m,n} = zeros(size(Ts1));
        Ts{m,n} = Ts1;
        RMs{m,n} = zeros(size(Ts{m,n}));
        f_max = zeros(8,8,10);
        for x = 1:8
            for y = 1:8
                for z = 1:10
                    % Remap the starting points and ending points of blocks
                    % according to the scale.
                    S1{x,y,z}.x = max([floor(S{x,y,z}.x*SC(m)*SC_t(n)),1]);
                    E1{x,y,z}.x = min(max(floor(E{x,y,z}.x*SC(m)*SC_t(n)),size(F_Ts,2)-size(F_Qs,2)+1),1);
                    S1{x,y,z}.y = max(floor(S{x,y,z}.y*SC(m)*SC_t(n)),1);
                    E1{x,y,z}.y = min(max(floor(E{x,y,z}.y*SC(m)*SC_t(n)),size(F_Ts,1)-size(F_Qs,1)+1),1);
                    if E1{x,y,z}.y < S1{x,y,z}.y
                        temp = S1{x,y,z}.y;
                        S1{x,y,z}.y = E1{x,y,z}.y;
                        E1{x,y,z}.y = temp;
                    end
                    if E1{x,y,z}.x < S1{x,y,z}.x
                        temp = S1{x,y,z}.x;
                        S1{x,y,z}.x = E1{x,y,z}.x;
                        E1{x,y,z}.x = temp;
                    end
                    if E{x,y,z}.z < S{x,y,z}.z
                        temp = S{x,y,z}.z;
                        S{x,y,z}.z = E1{x,y,z}.z;
                        E{x,y,z}.z = temp;
                    end
                    % Compute MCS only when the block is salient
                    if flags(x,y,z) == 1
                        for i = S1{x,y,z}.y:interval:E1{x,y,z}.y
                            for j = S1{x,y,z}.x:interval:E1{x,y,z}.x
                                for k = S{x,y,z}.z:E{x,y,z}.z
                                    if i+size(F_Qs,1)-1 <= size(F_Ts,1) && j+size(F_Qs,2)-1 <= size(F_Ts,2) ...
                                            && k + size(F_Qs,3)-1 <= size(F_Ts,3)
                                        
                                        F_T_i = F_Ts(i:space:i+size(F_Qs,1)-1, j:space:j+size(F_Qs,2)-1,k:space_t:k+size(F_Qs,3)-1,:);
                                        den = (norm_FQ*norm(F_T_i(:),'fro'));
                                        rho = F_Q1(:)'*F_T_i(:)/den;
                                        rho1 = F_Q2(:)'*F_T_i(:)/den;
                                        rho = max([rho rho1]);
                                        RMs{m,n}(floor(size(F_Qs,1)/2)+i,floor(size(F_Qs,2)/2)+j,floor(size(F_Qs,3)/2)+k) = (rho^2)/(1-rho^2);
                                    end
                                end
                            end
                        end
                        
                        
                        f = RMs{m,n}(floor(size(F_Qs,1)/2)+S1{x,y,z}.y:min(floor(size(F_Qs,1)/2)+E1{x,y,z}.y,size(RMs{m,n},1)),floor(size(F_Qs,2)/2)+S1{x,y,z}.x:min(floor(size(F_Qs,2)/2)+E1{x,y,z}.x,size(RMs{m,n},2)),...
                            min(floor(size(F_Qs,3)/2)+S{x,y,z}.z,size(RMs{m,n},3)):min(floor(size(F_Qs,3)/2)+E{x,y,z}.z,size(RMs{m,n},3)));
                        
                        % if the maximum resemblance value in the block is
                        % smaller than the threshold, do not search in the finer scale
                        
                        f_max(x,y,z) = max(f(:));
                        
                        if f_max(x,y,z) < 0.2
                            flags(x,y,z) = 0;
                        end
                        clear f;
                        
                    end
                end
            end
        end
        
        
        
    end
    % perform significance testing at each scale (controlling False
    % Discovery Rate. please check paper for more detail.)
    q = 0.1; % false discovery rate
    f_rho = RMs{length(SC),n}(:);
    %  [E_pdf,ind] = hist(f_rho(:),1000);
    [E_pdf, ind] = ksdensity(f_rho);
    E_cdf = cumsum(E_pdf/sum(E_pdf));
   
    p = 1-E_cdf; % p-values
    q = 0.1; %false discovery rate;
    
    p = sort(p(:));
    V = length(p);
    I = (1:V)';
    
    cVID = 1;
    cVN = sum(1./(1:V));
    
    pID = p(max(find(p<=I/V*q/cVID)));   
   
    detection = find(E_cdf>=1-pID);
    T_n = ind(detection(1)); %% Parameter for significance testing 2
    T_n
    if T_n < 0
        T_n = 0;
    end    
    
    for mm = 1:size(RMs{length(SC),n},3)
        [RM2,RM3] = stage3forMultiscale(RMs{length(SC),n}(:,:,mm),Qs(:,:,1),T_n);
        % rescale resemblance maps to the origianl target size
        RM2s(:,:,mm,n) = imresize(RM2,[size(F_T,1),size(F_T,2)],'nearest');
        RM3s(:,:,mm,n) = imresize(RM3,[size(F_T,1),size(F_T,2)],'nearest');
    end
    
end
disp(['Search time: ' num2str(toc) ' sec']);

