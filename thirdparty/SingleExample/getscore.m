X = zeros(0,1);
P = zeros(0,1);
st = 600;
j = st;

figure;

for i = 1:size(RV,3)
    sc(cat(3,RV(:,:,i),T(:,:,i)),'prob_jet'); colormap; c = caxis; P =[ P; c(2)];
    X = [X; j] ; j = j+1
    pause(0.1);
end

figure;
axes;
plot(X(20:end-20),P(20:end-20)),ylabel('RV score'),xlabel('Frame number'),title('Single Example method: query for action class 9 from Seq 1 on Seq 2')
hold on;
plot(linspace(621,672),0.2,'r.');
