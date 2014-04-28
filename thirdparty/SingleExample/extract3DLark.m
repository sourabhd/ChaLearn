
% Sourabh Daptardar :
% wrapper to extract 3D LARK features

clear all; close all; clc;
dbstop if error
if getenv('NSLOTS') == ''
	numSlots = 1;
else
	numSlots = str2num(getenv('NSLOTS'));
end
fprintf('Number of slots %d:', numSlots);
numThreads = feature('numThreads', numSlots);
fprintf('Num Threads: %d', numThreads);

dumpfile = 'Dump_3DLark.mat'

%% Parameters for 3D LARK
param.wsize = 3; % LARK spatial window size
param.wsize_t = 7; % LARK temporal window size
param.alpha = 0.29; % LARK sensitivity parameter
param.h = 1;  % smoothing parameter for LARK
param.sigma = 0.01; % fall-off parameter for self-resemblamnce
param.interval = 1  ;
param.resol = [64,64]


compute = 1;

Q_VideoSeqNum = [1,2];
Q_NumSeq = size(Q_VideoSeqNum,2);
for i = 1:Q_NumSeq
	Q_csv_fl{i} = sprintf('../../dense_trajectory/training/train/Seq0%d_labels.csv', i)
	Q_action_labels{i} = csvread(Q_csv_fl{i})
end

%matlabpool('open',numThreads)
for i = 1:Q_NumSeq
	clear AL;
	AL = Q_action_labels{i};
	Q_NumQ = size(AL,1);
	for j = 1:Q_NumQ
		Q_videoFname{i,j} = sprintf('../../dense_trajectory/trainingVideos/Seq0%d/output_Seq0%d_%d_frame%d-%d.avi',i,i,AL(j,1),AL(j,2),AL(j,3));
		disp(Q_videoFname{i,j});
		Q_seq_row_to_labels{i,j} = AL(j,:);
		%Q_seq_labels_to_row{i,AL(j,1),AL(j,2),AL(j,3)} = j;
		Q_video{i,j} = VideoReader(Q_videoFname{i, j});
		Q_videoFrames{i,j} = read(Q_video{i,j});
		%videoHeight, videoWidth, videoChannels, videoNumFrames = size(videoFrames);
		Q_videoFrames_sz{i, j} = size(Q_videoFrames{i,j})
	end
end
%matlabpool close


%%sampling = 3;


%matlabpool('open',numThreads)
Q = cell(Q_NumSeq,Q_NumQ);
Y_Q = cell(Q_NumSeq,Q_NumQ);
Seq_Q = cell(Q_NumSeq,Q_NumQ);
W_Q = cell(Q_NumSeq,Q_NumQ);
QueryF = cell(Q_NumSeq,Q_NumQ);
meanLSK = cell(Q_NumSeq,Q_NumQ);
vs = cell(Q_NumSeq,Q_NumQ);


for i = 1:Q_NumSeq
	clear AL;
	AL = Q_action_labels{i};
	Q_NumQ = size(AL,1);
	for j = 1:Q_NumQ
		sampling = 1;
		begin = 1;
		stop = Q_videoFrames_sz{i,j}(4);
		for k = begin:stop
			Y_Q{i,j}(:,:,k-begin+1) = double(rgb2gray(Q_videoFrames{i,j}(:,:,:,k)));
			Seq_Q{i,j}(:,:,k-begin+1) = imresize(Y_Q{i, j}(:,:,k-begin+1),param.resol,'bilinear');
		end
       Seq_Q{i,j} = Seq_Q{i,j}/std(Seq_Q{i,j}(:));

		Q{i, j} = smooth3(Y_Q{i, j},'gaussian',[7,7,1]);
		Q{i, j} = Q{i, j}(1:sampling:end,1:sampling:end,:);
		Q{i, j} = smooth3(Q{i,j},'gaussian',[7,7,1]);

	end
end
%matlabpool close

save(dumpfile, '-v7.3', 'param', 'Q', 'Seq_Q', 'Y_Q', 'Q_NumSeq', 'Q_action_labels', 'Q_videoFrames', 'Q_seq_row_to_labels');

%% Compute 3-D LARKs for query


parfor i = 1:Q_NumSeq
	clear AL;
	AL = Q_action_labels{i};
	Q_NumQ = size(AL,1);
	for j = 1:Q_NumQ
		tic;
		W_Q{i, j} = ThreeDLARK1(Q{i, j},param);
		disp(['Q: 3-D LARK computation : ' num2str(toc) 'sec']);
		tic;
		% PCA projection of query
		[ QueryF{i,j}, meanLSK{i, j}, vs{i, j}] = PCAfeature(param.wsize,param.wsize_t,W_Q{i, j});

		disp(['PCA : ' num2str(i) '-' num2str(j) ' : '  num2str(toc) 'sec']);
	end
end
save(dumpfile, '-v7.3', '-append','W_Q', 'QueryF', 'meanLSK', 'vs');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%if compute == 1
%	targetVideoFname = '../../dense_trajectory/training/train/Seq02_color.mp4'
%	targetVideo = VideoReader(targetVideoFname);
%	targetVideoFrames = read(targetVideo);
%	fprintf('Size of target :'); size(targetVideoFrames)
%	sz = size(targetVideoFrames)
%
%	%begin = 1;
%	%stop = 10;
%	%stop = 456;%580;
%	%stop = sz(4)
%	begin = 600
%	stop = 850
%	for i = begin:stop
%		Y(:,:,i-begin+1) = double(rgb2gray(targetVideoFrames(:,:,:,i)));
%		Seq(:,:,i-begin+1) = imresize(Y(:,:,i-begin+1),[64 64],'bilinear');
%	end
%	Seq = Seq/std(Seq(:));
%
%	T = smooth3(Y,'gaussian',[7,7,1]);
%	T = T(1:sampling:end,1:sampling:end,:);
%	T = smooth3(T,'gaussian',[7,7,1]);
%
%end
%
%if compute == 1
%	% Compute 3-D LARKs for target
%	tic;
%	W_T = ThreeDLARK1(T,param);
%	% PCA projection of target
%	for i = 1:size(W_T,1)
%		for j = 1:size(W_T,2)
%			for k = 1:size(W_T,3)
%				Ws = W_T(i,j,k,:);
%				TargetF(i,j,k,:) = ((Ws(:)  - meanLSK)'*vs)';
%			end
%		end
%	end
%
%	disp(['T: 3-D LARK computation : ' num2str(toc) 'sec']);
%
%	save(dumpfile, '-v7.3','-append')
%end
%
%if compute == 1
%	%% generate resemblance volume using query and its mirror-reflected version
%	%% (single scale)
%	RV = GenerateRVs_both(QueryF,TargetF);
%
%	save(dumpfile, '-v7.3','-append')
%end
%
%if compute == 1
%	f3 = figure(3)
%	set(f3,'renderer','zbuffer');
%	set(f3,'visible','off');
%	aviobj3 = VideoWriter('RV.avi');
%	aviobj3.FrameRate = 15;
%	open(aviobj3);
%	for i = 1:size(RV,3)
%		i
%		movegui(f3,'onscreen');
%		sc(cat(3,RV(:,:,i),T(:,:,i)),'prob_jet'); colorbar; pause(0.1); frame = getframe(f3); writeVideo(aviobj3,frame);
%
%	end
%	close(aviobj3);
%	save(dumpfile, '-v7.3','-append')
%end
%
%%% Space-time saliency
%% Compute space-time saliency
%tic;
%LARK = ThreeDLARK1(Seq,param);
%SM = SpaceTimeSaliencyMap(Seq,LARK,param.wsize,param.wsize_t,param.sigma,T);
%disp(['SpaceTimeSaliencyMap : ' num2str(toc) 'sec']);
%[block,flag,S,E] = Proto_Action(SM,[8,8,10],0.5);
%aviobj1 = VideoWriter('SpaceTimeSaliencyMap.avi');
%aviobj1.FrameRate = 15;
%open(aviobj1)
%f1 = figure(1),
%set(f1,'visible','off');
%set(f1,'renderer','zbuffer');
%for i = 1:size(Seq,3)
%	i
%	movegui(f1,'onscreen');
%	a = imresize(SM(:,:,i),[size(Y,1) size(Y,2)]);
%	b = imresize(block(:,:,i),[size(Y,1) size(Y,2)]);
%	subplot(1,2,1),sc(cat(3,a, Y(:,:,i)),'prob_jet',[min(SM(:)) max(SM(:))]); colorbar;
%	subplot(1,2,2),sc(cat(3,b, Y(:,:,i)),'prob_jet');
%	pause(.01);
%	frame = getframe(f1);
%	writeVideo(aviobj1,frame);
%end
%close(aviobj1);
%
%save(dumpfile, '-v7.3','-append')
%
%%% Multi-scale approach
%[RMs,Ts,RM2s,RM3s,SC_t] = MultiScaleSearch_CoarseToFine(Q,T,QueryF,TargetF,S,E,flag);
%
%%% Significance testing (controlling False Discovery Rate. please check paper for more detail.)
%
%E_RM = max(RM2s,[],4);
%[E_RM1,s_ind] = max(RM3s,[],4); %ML estimation of scale
%f_rho = E_RM1;
%[E_pdf, ind] = ksdensity(f_rho(:));
%E_cdf = cumsum(E_pdf/sum(E_pdf));
%
%%     figure, plot(ind,E_pdf/sum(E_pdf));
%
%
%p = 1-E_cdf; % p-values
%q = 0.1; %false discovery rate;
%
%p = sort(p(:));
%V = length(p);
%I = (1:V)';
%
%cVID = 1;
%cVN = sum(1./(1:V));
%
%pID = p(max(find(p<=I/V*q/cVID)));
%
%figure, plot(ind,p);
%hold on; plot(ind,I/V*q/cVID,'r');
%
%detection = find(E_cdf>=1-pID);
%
%
%T_n = ind(detection(1));
%if T_n < 0
%	T_n = 0;
%end
%
%aviobj = VideoWriter('Detection.avi');
%aviobj.FrameRate = 15;
%open(aviobj)
%f = figure(1),
%set(f,'renderer','zbuffer')
%set(f,'visible','off')
%for mm = 1:size(E_RM,3)
%	[E_RM2(:,:,mm),RM3(:,:,mm)] = FinalStage3(E_RM1(:,:,mm),Q(:,:,1),T(:,:,1),T_n,s_ind(:,:,mm),SC_t,1);
%	subplot(1,2,1), sc(cat(3,imresize(E_RM(:,:,mm),[size(T,1),size(T,2)]),T(:,:,mm)),'prob_jet'); colorbar;
%	subplot(1,2,2), sc(cat(3,imresize(E_RM2(:,:,mm),[size(T,1),size(T,2)]),T(:,:,mm)),'prob_jet'); colorbar;
%	pause(.1);
%	frame = getframe(f);
%	writeVideo(aviobj,frame);
%end
%close(aviobj)
%save(dumpfile, '-v7.3','-append')

