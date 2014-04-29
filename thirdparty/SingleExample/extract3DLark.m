
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

if true
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
else
	load(dumpfile);
end

%% Compute 3-D LARKs for query

if true
	for i = 1:Q_NumSeq
		clear AL;
		AL = Q_action_labels{i};
		Q_NumQ = size(AL,1);
		parfor j = 1:Q_NumQ
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
else
	load(dumpfile);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if true
	targetVideoFname = '../../dense_trajectory/training/train/Seq02_color.mp4'
	targetVideo = VideoReader(targetVideoFname);
	targetVideoFrames = read(targetVideo);
	fprintf('Size of target :'); size(targetVideoFrames)
	sz = size(targetVideoFrames)

	begin = 1;
	%stop = 10;
	%stop = 456;%580;
	stop = sz(4)
	%begin = 600
	%stop = 850
	for i = begin:stop
		Y(:,:,i-begin+1) = double(rgb2gray(targetVideoFrames(:,:,:,i)));
		Seq(:,:,i-begin+1) = imresize(Y(:,:,i-begin+1),[64 64],'bilinear');
	end
	Seq = Seq/std(Seq(:));

	T = smooth3(Y,'gaussian',[7,7,1]);
	T = T(1:sampling:end,1:sampling:end,:);
	T = smooth3(T,'gaussian',[7,7,1]);

	save(dumpfile, '-v7.3', '-append', 'T', 'Seq', 'Y');
else
	load(dumpfile);
end


if true 
	% Compute 3-D LARKs for target
	tic;
	W_T = ThreeDLARK1(T,param);
	% PCA projection of target
	for i = 1:size(W_T,1)
		for j = 1:size(W_T,2)
			for k = 1:size(W_T,3)
				Ws = W_T(i,j,k,:);
				for x = 1:Q_NumSeq
					parfor y = 1:Q_NumQ
						TargetF{x,y}(i,j,k,:) = ((Ws(:)  - meanLSK{x,y})'*vs{x,y})';
					end
				end
			end
		end
	end

	disp(['T: 3-D LARK computation : ' num2str(toc) 'sec']);

	save(dumpfile, '-v7.3', '-append','W_T', 'Ws', 'TargetF');
else
	load(dumpfile);
end

if true
	%% generate resemblance volume using query and its mirror-reflected version
	%% (single scale)
	for x = 1:Q_NumSeq
		parfor y = 1:Q_NumQ
			RV{x,y} = GenerateRVs_both(QueryF{x,y},TargetF{x,y});
		end
	end

	save(dumpfile, '-v7.3','-append','RV')
else
	load(dumpfile);
end

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
%% Space-time saliency
% Compute space-time saliency

if true
	tic;
	LARK = ThreeDLARK1(Seq,param);
	SM = SpaceTimeSaliencyMap(Seq,LARK,param.wsize,param.wsize_t,param.sigma,T);
	disp(['SpaceTimeSaliencyMap : ' num2str(toc) 'sec']);
	[block,flag,S,E] = Proto_Action(SM,[8,8,10],0.5);
	save(dumpfile, '-v7.3','-append','LARK','SM','block','flag','S','E');
else
	load(dumpfile);
end

%aviobj1 = VideoWriter('SpaceTimeSaliencyMap.avi');
%aviobj1.FrameRate = 15;
%open(aviobj1)
%f1 = figure(1),
%set(f1,'visible','off');
%set(f1,'renderer','zbuffer');
%for i = 1:size(Seq,3)
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


%% Multi-scale approach
if true
	for x = 1:Q_NumSeq
		parfor y = 1:Q_NumQ
			[RMs{x,y},Ts{x,y},RM2s{x,y},RM3s{x,y},SC_t{x,y}] = MultiScaleSearch_CoarseToFine(Q{x,y},T,QueryF{x,y},TargetF{x,y},S,E,flag);

			%% Significance testing (controlling False Discovery Rate. please check paper for more detail.)

			E_RM{x,y} = max(RM2s{x,y},[],4);
			[E_RM1{x,y},s_ind{x,y}] = max(RM3s{x,y},[],4); %ML estimation of scale
			f_rho{x,y} = E_RM1{x,y};
			[E_pdf{x,y}, ind{x,y}] = ksdensity(f_rho{x,y}(:));
			E_cdf{x,y} = cumsum(E_pdf{x,y}/sum(E_pdf{x,y}));

			%     figure, plot(ind,E_pdf/sum(E_pdf));


			p{x,y} = 1-E_cdf{x,y}; % p-values
			q{x,y} = 0.1; %false discovery rate;

			p{x,y} = sort(p{x,y}(:));
			V{x,y} = length(p{x,y});
			I{x,y} = (1:V{x,y})';

			cVID = 1;
			cVN{x,y} = sum(1./(1:V{x,y}));

			pID{x,y} = p{x,y}(max(find(p<=I{x,y}/V{v,y}*q{x,y}/cVID)));

	%		figure, plot(ind,p);
	%		hold on; plot(ind,I/V*q/cVID,'r');

			detection{x,y} = find(E_cdf{x,y}>=1-pID{x,y});


			T_n{x,y} = ind{x,y}(detection{x,y}(1));
			if T_n{x,y} < 0
				T_n{x,y} = 0;
			end

		end
	end

	save(dumpfile, '-v7.3','-append','RMs','Ts','RM2s','RM3s','SC_t','E_RM','E_RM1','s_ind','f_rho','E_pdf','ind','E_cdf','p','q','V','I','cVN','pID','detection','T_n');

else
	load(dumpfile);
end

%
%aviobj = VideoWriter('Detection.avi');
%aviobj.FrameRate = 15;
%open(aviobj)
%f = figure(1),
%set(f,'renderer','zbuffer')
%set(f,'visible','off')

if true
	for x = 1:Q_NumSeq
		parfor y = 1:Q_NumQ
			for mm = 1:size(E_RM{x,y},3)
					[E_RM2{x,y}(:,:,mm),RM3{x,y}(:,:,mm)] = FinalStage3(E_RM1{x,y}(:,:,mm),Q{x,y}(:,:,1),T(:,:,1),T_n{x,y},s_ind{x,y}(:,:,mm),SC_t{x,y},1);
				%	subplot(1,2,1), sc(cat(3,imresize(E_RM(:,:,mm),[size(T,1),size(T,2)]),T(:,:,mm)),'prob_jet'); colorbar;
				%	subplot(1,2,2), sc(cat(3,imresize(E_RM2(:,:,mm),[size(T,1),size(T,2)]),T(:,:,mm)),'prob_jet'); colorbar;
				%	pause(.1);
				%	frame = getframe(f);
				%	writeVideo(aviobj,frame);
			end
		end
	end
	%close(aviobj)
	%save(dumpfile, '-v7.3','-append')
	save(dumpfile, '-v7.3','-append','E_RM2','RM3');
else
	load(dumpfile)
end
