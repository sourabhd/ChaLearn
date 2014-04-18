% action detection Demo
% [HISTORY]
% June 1, 2011 : created by Hae Jong Seo

clear all;
close all;
clc;

%% Read Query & Target

sampling = 3;
load('BeachWalk_query.mat');
Q = smooth3(Q,'gaussian',[7,7,1]);
Q = Q(1:sampling:end,1:sampling:end,:);
Q = smooth3(Q,'gaussian',[7,7,1]);

begin = 1;
stop = 10;
%stop = 456;%580;
for i = begin:stop
    Y(:,:,i-begin+1) = double(imread(['Beach/Beach_data' num2str(i) '.jpg']));
    Seq(:,:,i-begin+1) = imresize(Y(:,:,i-begin+1),[64 64],'bilinear');
end
Seq = Seq/std(Seq(:));

T = smooth3(Y,'gaussian',[7,7,1]);
T = T(1:sampling:end,1:sampling:end,:);
T = smooth3(T,'gaussian',[7,7,1]);

%% Parameters for 3D LARK
param.wsize = 3; % LARK spatial window size
param.wsize_t = 7; % LARK temporal window size
param.alpha = 0.29; % LARK sensitivity parameter
param.h = 1;  % smoothing parameter for LARK
param.sigma = 0.01; % fall-off parameter for self-resemblamnce
param.interval = 1  ;

% Compute 3-D LARKs for query
tic;
W_Q = ThreeDLARK1(Q,param);
disp(['Q: 3-D LARK computation : ' num2str(toc) 'sec']);
tic;
% PCA projection of query
[QueryF,meanLSK,vs] = PCAfeature(param.wsize,param.wsize_t,W_Q);
disp(['PCA : ' num2str(toc) 'sec']);

% Compute 3-D LARKs for target
tic;
W_T = ThreeDLARK1(T,param);
% PCA projection of target
for i = 1:size(W_T,1)
    for j = 1:size(W_T,2)
        for k = 1:size(W_T,3)
            Ws = W_T(i,j,k,:);
            TargetF(i,j,k,:) = ((Ws(:)  - meanLSK)'*vs)';
        end
    end
end

disp(['T: 3-D LARK computation : ' num2str(toc) 'sec']);

%% generate resemblance volume using query and its mirror-reflected version
%% (single scale)
RV = GenerateRVs_both(QueryF,TargetF);

figure(3)
for i = 1:size(RV,3)
    sc(cat(3,RV(:,:,i),T(:,:,i)),'prob_jet'); colorbar; pause(0.1);
end

%% Space-time saliency
% Compute space-time saliency
tic;
LARK = ThreeDLARK1(Seq,param);
SM = SpaceTimeSaliencyMap(Seq,LARK,param.wsize,param.wsize_t,param.sigma,T);
disp(['SpaceTimeSaliencyMap : ' num2str(toc) 'sec']);
[block,flag,S,E] = Proto_Action(SM,[8,8,10],0.5);
figure(1),
for i = 1:size(Seq,3)
    a = imresize(SM(:,:,i),[size(Y,1) size(Y,2)]);
    b = imresize(block(:,:,i),[size(Y,1) size(Y,2)]);
    subplot(1,2,1),sc(cat(3,a, Y(:,:,i)),'prob_jet',[min(SM(:)) max(SM(:))]); colorbar;
    subplot(1,2,2),sc(cat(3,b, Y(:,:,i)),'prob_jet');
    pause(.01);
end

%% Multi-scale approach
[RMs,Ts,RM2s,RM3s,SC_t] = MultiScaleSearch_CoarseToFine(Q,T,QueryF,TargetF,S,E,flag);

%% Significance testing (controlling False Discovery Rate. please check paper for more detail.)

E_RM = max(RM2s,[],4);
[E_RM1,s_ind] = max(RM3s,[],4); %ML estimation of scale
f_rho = E_RM1;
[E_pdf, ind] = ksdensity(f_rho(:));
E_cdf = cumsum(E_pdf/sum(E_pdf));

%     figure, plot(ind,E_pdf/sum(E_pdf));


p = 1-E_cdf; % p-values
q = 0.1; %false discovery rate;

p = sort(p(:));
V = length(p);
I = (1:V)';

cVID = 1;
cVN = sum(1./(1:V));

pID = p(max(find(p<=I/V*q/cVID)));

figure, plot(ind,p);
hold on; plot(ind,I/V*q/cVID,'r');

detection = find(E_cdf>=1-pID);


T_n = ind(detection(1));
if T_n < 0
    T_n = 0;
end
figure(1),
for mm = 1:size(E_RM,3)
    [E_RM2(:,:,mm),RM3(:,:,mm)] = FinalStage3(E_RM1(:,:,mm),Q(:,:,1),T(:,:,1),T_n,s_ind(:,:,mm),SC_t,1);
    subplot(1,2,1), sc(cat(3,imresize(E_RM(:,:,mm),[size(T,1),size(T,2)]),T(:,:,mm)),'prob_jet'); colorbar;
    subplot(1,2,2), sc(cat(3,imresize(E_RM2(:,:,mm),[size(T,1),size(T,2)]),T(:,:,mm)),'prob_jet'); colorbar;
    pause(.1);
end
