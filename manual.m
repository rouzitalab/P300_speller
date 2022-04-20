clc;
clearvars;

fprintf(1, 'Performing classification on my data by my LDA... \n\n' );
[pathstr,~,~] = fileparts(which(mfilename));
d1=[pathstr,'\last\SubjectD.mat'];
load(d1);
window=320; % window after stimulus (1s)
channel=[1 2 3 4 5 6 7 8 9 10]; % using Fz, C3, Cz(11), C4, P3, Pz, P4, PO7, PO8, Oz for analysis and plots

% convert to double precision
Sig=double(Sig);
StimulusCode=double(StimulusCode);
StimulusType=double(StimulusType);


% % FIR Filtering
[b,a]=butter(5,[.01 16]/(120),'bandpass');
Signal=filter(b,a,Signal);

% Outlier Removing
for i = 1 : size(sig,3)
    for j = 1 : size(sig,1)
        m = mean( sig(j,:,i) );
        mx = max( sig(j,:,i) );
        mn = min( sig(j,:,i) );
        ind = sig(j,:,i) < m ;
        q1 = mean( sig(j,ind,i) );
        ind = sig(j,:,i) > m ;
        q3 = mean( sig(j,ind,i) );
        iqr = q3 - q1;
        mxOutlr = q3 + iqr;
        mnOutlr = q1 - iqr;
        ind = sig(j,:,i) < mnOutlr ;
        sig(j,ind,i) = m;
        ind = sig(j,:,i) > mxOutlr ;
        sig(j,ind,i) = m;
    end
end

% 6 X 6 onscreen matrix
screen=char('A','G','M','S','Y','5',...
            'B','H','N','T','Z','6',...
            'C','I','O','U','1','7',...
            'D','J','P','V','2','8',...
            'E','K','Q','W','3','9',...
            'F','L','R','X','4','_');

% for each character epoch
for epoch=1:size(sig,1)
    
    % get reponse samples
    count=1;
    rowcolcnt=ones(1,12);
    for n=2:size(sig,2)
        if Flashing(epoch,n)==0 & Flashing(epoch,n-1)==1
            rowcol=StimulusCode(epoch,n-1);
            responses(rowcol,rowcolcnt(rowcol),:,:)=sig(epoch,n:n+window-1,:);
            rowcolcnt(rowcol)=rowcolcnt(rowcol)+1;
        end
    end

    % average and group responses by letter
    m=1;
    avgresp=mean(responses,2);
    avgresp=reshape(avgresp,12,window,64);
    for row=7:12
        for col=1:6
            letter(epoch,m,:,:)=(avgresp(row,:,:)+avgresp(col,:,:))/2;
            % row-column intersection
            letter(m,:,:)=(avgresp(row,:,:)+avgresp(col,:,:))/2;
            % the crude avg peak classifier score (**tuned for Subject_A**)
            score(m)=mean(letter(m,30:100,channel(1)))-mean(letter(m,110:150,channel(1)));
            m=m+1;
        end
    end
    
    [val,index]=max(score);
    charvect(epoch)=screen(index);
    
    % if labeled, get target label and response
    if isempty(StimulusType)==0
        label=unique(StimulusCode(epoch,:).*StimulusType(epoch,:));
        targetlabel=(6*(label(3)-7))+label(2);
        Target(epoch,:,:)=.5*(avgresp(label(2),:,:)+avgresp(label(3),:,:));
        NonTarget(epoch,:,:)=mean(avgresp,1)-(1/6)*Target(epoch,:,:);
    end
    
    sig(epoch,:,:,:,:) = responses;
    
end
stmTypeAvg = zeros(85,12);
for i = 1 : 85
    for j = 1 : 12
        if TargetChar(i) == 'A'
            stmTypeAvg(i,1) = 1;
            stmTypeAvg(i,7) = 1;
        elseif TargetChar(i) == 'B'
            stmTypeAvg(i,2) = 1;
            stmTypeAvg(i,7) = 1;
        elseif TargetChar(i) == 'C'
            stmTypeAvg(i,3) = 1;
            stmTypeAvg(i,7) = 1;
        elseif TargetChar(i) == 'D'
            stmTypeAvg(i,4) = 1;
            stmTypeAvg(i,7) = 1;
        elseif TargetChar(i) == 'E'
            stmTypeAvg(i,5) = 1;
            stmTypeAvg(i,7) = 1;
        elseif TargetChar(i) == 'F'
            stmTypeAvg(i,6) = 1;
            stmTypeAvg(i,7) = 1;
        elseif TargetChar(i) == 'G'
            stmTypeAvg(i,1) = 1;
            stmTypeAvg(i,8) = 1;
        elseif TargetChar(i) == 'H'
            stmTypeAvg(i,2) = 1;
            stmTypeAvg(i,8) = 1;
        elseif TargetChar(i) == 'I'
            stmTypeAvg(i,3) = 1;
            stmTypeAvg(i,8) = 1;
        elseif TargetChar(i) == 'J'
            stmTypeAvg(i,4) = 1;
            stmTypeAvg(i,8) = 1;
        elseif TargetChar(i) == 'K'
            stmTypeAvg(i,5) = 1;
            stmTypeAvg(i,8) = 1;
        elseif TargetChar(i) == 'L'
            stmTypeAvg(i,6) = 1;
            stmTypeAvg(i,8) = 1;
        elseif TargetChar(i) == 'M'
            stmTypeAvg(i,1) = 1;
            stmTypeAvg(i,9) = 1;
        elseif TargetChar(i) == 'N'
            stmTypeAvg(i,2) = 1;
            stmTypeAvg(i,9) = 1;
        elseif TargetChar(i) == 'O'
            stmTypeAvg(i,3) = 1;
            stmTypeAvg(i,9) = 1;
        elseif TargetChar(i) == 'P'
            stmTypeAvg(i,4) = 1;
            stmTypeAvg(i,9) = 1;
        elseif TargetChar(i) == 'Q'
            stmTypeAvg(i,5) = 1;
            stmTypeAvg(i,9) = 1;
        elseif TargetChar(i) == 'R'
            stmTypeAvg(i,6) = 1;
            stmTypeAvg(i,9) = 1;
        elseif TargetChar(i) == 'S'
            stmTypeAvg(i,1) = 1;
            stmTypeAvg(i,10) = 1;
        elseif TargetChar(i) == 'T'
            stmTypeAvg(i,2) = 1;
            stmTypeAvg(i,10) = 1;
        elseif TargetChar(i) == 'U'
            stmTypeAvg(i,3) = 1;
            stmTypeAvg(i,10) = 1;
        elseif TargetChar(i) == 'V'
            stmTypeAvg(i,4) = 1;
            stmTypeAvg(i,10) = 1;
        elseif TargetChar(i) == 'W'
            stmTypeAvg(i,5) = 1;
            stmTypeAvg(i,10) = 1;
        elseif TargetChar(i) == 'X'
            stmTypeAvg(i,6) = 1;
            stmTypeAvg(i,10) = 1;
        elseif TargetChar(i) == 'Y'
            stmTypeAvg(i,1) = 1;
            stmTypeAvg(i,11) = 1;
        elseif TargetChar(i) == 'Z'
            stmTypeAvg(i,2) = 1;
            stmTypeAvg(i,11) = 1;
        elseif TargetChar(i) == '1'
            stmTypeAvg(i,3) = 1;
            stmTypeAvg(i,11) = 1;
        elseif TargetChar(i) == '2'
            stmTypeAvg(i,4) = 1;
            stmTypeAvg(i,11) = 1;
        elseif TargetChar(i) == '3'
            stmTypeAvg(i,5) = 1;
            stmTypeAvg(i,11) = 1;
        elseif TargetChar(i) == '4'
            stmTypeAvg(i,6) = 1;
            stmTypeAvg(i,11) = 1;
        elseif TargetChar(i) == '5'
            stmTypeAvg(i,1) = 1;
            stmTypeAvg(i,12) = 1;
        elseif TargetChar(i) == '6'
            stmTypeAvg(i,2) = 1;
            stmTypeAvg(i,12) = 1;
        elseif TargetChar(i) == '7'
            stmTypeAvg(i,3) = 1;
            stmTypeAvg(i,12) = 1;
        elseif TargetChar(i) == '8'
            stmTypeAvg(i,4) = 1;
            stmTypeAvg(i,12) = 1;
        elseif TargetChar(i) == '9'
            stmTypeAvg(i,5) = 1;
            stmTypeAvg(i,12) = 1;
        elseif TargetChar(i) == '_'
            stmTypeAvg(i,6) = 1;
            stmTypeAvg(i,12) = 1;
        end
    end
end
avgSig = mean(sig,3);
avgSig = reshape(avgSig,85,12,window,64);
testSize = 15;
scr = zeros(40,36);
trialSize = 25;
letr = [1 2 3 4 5 6;
        7 8 9 10 11 12;
        13 14 15 16 17 18;
        19 20 21 22 23 24;
        25 26 27 28 29 30;
        31 32 33 34 35 36];
    acc = 0;
for k = 41 : 85
    rowColScr = zeros(12,1);
end

for tepoch = 1 : size(channel,2) 
% trainTarg = zeros(1200,160);
% trainNonTarg = zeros(6000,160);
indTarg = 1;
indNonTarg = 1;
for i = 1 : trialSize
    for j = 1 : 12
        if StimulusType(i,j)==1
            trainTarg(indTarg:indTarg+9,(window*(tepoch-1))+1:window*tepoch) = Sig(i,j,:,:,channel(tepoch));
            indTarg = indTarg + 10;
        else
            trainNonTarg(indNonTarg:indNonTarg+9,(window*(tepoch-1))+1:window*tepoch) = Sig(i,j,:,:,channel(tepoch));
            indNonTarg = indNonTarg + 10;
        end
    end
end
end
tmp = downsample(trainTarg',16);
trainTarg = tmp';
tmp = downsample(trainNonTarg',16);
trainNonTarg = tmp';
W = LDATrain(trainTarg,trainNonTarg);
vote = zeros(testSize,12);
for i = trialSize+1 : size(Sig,1)
    for j = 1 : 12
        for ch = 1 : size(channel,2)
            testData(:,(window*(ch-1))+1:window*ch) = Sig(i,j,:,:,channel(ch));
        end
        tmp = downsample(testData',16);
        testData = tmp';
        vote(i-trialSize,j) = LDATest(W,testData);
    end
end
for k = 1 : testSize
[maxcol,col] = max(vote(k,7:12));
[maxrow,row] = max(vote(k,1:6));
indx = (col-1)*6 + row;
fprintf(1, 'Epoch: %d  Predicted: %c Target: %c\n',k+trialSize,screen(indx),TargetChar(k+trialSize));
if screen(indx) == TargetChar(k+trialSize)
    acc = acc + 1;
end
end

fprintf(1, '\nAccuracy = %d\n\n' ,(acc*100/testSize));

trainData = [trainTarg; trainNonTarg];
group = [ones(1,size(trainTarg,1)), zeros(1,size(trainNonTarg,1))];
indTest = 1;
for i = 66 : 85
    for j = 1 : 12
        testData(indTest,:) = avgSig(i,j,:,channel(tepoch));
        indTest = indTest + 1;
    end
end
tmp = downsample(trainData',8);
trainData = tmp';
for j = 1 : 12
testData = zeros(15,160);
for i = 1 : 15
    testData(i,:) = sig(k,j,i,:,channel(tepoch));
end
tmp = downsample(testData',8);
testData = tmp';
svmStruct = svmtrain(trainData, group);
class1 = svmclassify(svmStruct, testData);
W = LDA (trainData, group);
class1 = testData * W';
class1 = classify (testData, trainData, group);
class1 = knnclassify (testData, trainData, group');
rowColScr(j) = rowColScr(j) + sum(class1);
end
for i = 0 : (testSize - 1)
    for j = 1 : 6
        for k = 7 : 12
            if class1(12*i+j) == 1 & class1(12*i+k) == 1
                scr(i+1,letr(j,k-6)) = scr(i+1,letr(j,k-6)) + 1;
            end
        end
    end
end
    
end

acc = 0;
for i = 1 : testSize
    [val, indx] = max(scr(i,:));

end

%  display results

if isempty(TargetChar)==0

    k=0;
    for p=1:size(Signal,1)
        if charvect(p)==TargetChar(p)
            k=k+1;
        end
    end

    correct=(k/size(Signal,1))*100;

    fprintf(1, 'Classification Results: \n\n' );
    for kk=1:size(Signal,1)
        fprintf(1, 'Epoch: %d  Predicted: %c Target: %c\n',kk,charvect(kk),TargetChar(kk));
    end
    fprintf(1, '\n %% Correct from Labeled Data: %2.2f%% \n',correct);

    % plot averaged responses and topography
    Tavg=reshape(mean(Target(:,:,:),1),window,64);
    NTavg=reshape(mean(NonTarget(:,:,:),1),window,64);
    figure
    plot([1:window]/window,Tavg(:,channel),'linewidth',2)
    hold on
    plot([1:window]/window,NTavg(:,channel),'r','linewidth',2)
    title('Averaged P300 Responses over Cz')
    legend('Targets','NonTargets');
    xlabel('time (s) after stimulus')
    ylabel('amplitude (uV)')
    
    % Target/NonTarget voltage topography plot at 300ms (sample 72)
    vdiff=abs(Tavg(72,:)-NTavg(72,:));
    figure
    topoplotEEG(vdiff,'eloc64.txt','gridscale',150)
    title('Target/NonTarget Voltage Difference Topography at 300ms')
    caxis([min(vdiff) max(vdiff)])
    colorbar
    
else

    for kk=1:size(Signal,1)
        fprintf(1, 'Epoch: %d  Predicted: %c\n',kk,charvect(kk));
    end

end

fprintf(1, '\nThe resulting classified character vector is the variable named "charvect". \n');
fprintf(1, 'This is an example of how the results *must* be formatted for submission. \n');
fprintf(1, 'The character vectors from each case and subject are to be labeled, grouped, and submitted according to the accompanied documentation. \n');


% convert to double precision
Signal=double(Signal);
Flashing=double(Flashing);
StimulusCode=double(StimulusCode);
StimulusType=double(StimulusType);





%% Data Manipulation
chList = [9 11 13 34 51 56 60];
stmType = zeros(85,180);
tmp = Signal(:,:,chList);
sigBloc = zeros(85,28800,7);
for i = 1 : size(sigBloc,3)
    for j = 1 : size(sigBloc,1)
        for k = 0 : 179
            sigBloc(j,160*k+1:160*(k+1),i) = tmp(j,42*k+1:42*k+160,i);
        end
    end
end
for j = 1 : 85
    for i = 0 : 179
        if StimulusType(j,42*i+1) == 1
            stmType(j,i+1) = 1;
        end
    end
end

stmCode = zeros (85, 180);
for i = 1 : 85
    for j = 0 : 179
        stmCode(i,j+1) = StimulusCode(i,42*j + 1);
    end
end

%% Averaging Through Trials
avrgSig = zeros(85,1920,7);
for l = 1 : 7
    for i = 1 : 85
        for j = 1 : 12
            indices = find(stmCode(i,:)==j);
            for k = 1 : 15
                tmp = sigBloc (i, 160*(indices(k)-1)+1: 160*indices(k), l);
                avrgSig(i,160*(j-1)+1:160*j,l) = avrgSig(i,160*(j-1)+1:160*j,l) + tmp;
            end
            avrgSig(i,160*(j-1)+1:160*j,l) = avrgSig(i,160*(j-1)+1:160*j,l) / 15;
        end
    end
end


%% P300 and Non-P300 Seperation in Training Data (First 60 Epochs)
sigP3 = zeros(60,320,7);
sigNonP3 = zeros(60,1600,7);
sigP3Avrg = zeros(1,160);
sigNonP3Avrg = zeros(1,160);
for i = 1 : 60
    p300Ind = 0;
    nonP300Ind = 0;
    for j = 0 : 11
        if stmTypeAvrg(i,j+1) == 1
            sigP3(i,160*p300Ind + 1 : 160*(p300Ind+1),:) = avrgSig(i,160*j + 1 : 160*(j+1), :);
            p300Ind = p300Ind + 1;
        else
            sigNonP3(i,160*nonP300Ind + 1 : 160*(nonP300Ind+1),:) = avrgSig(i,160*j + 1 : 160*(j+1), :);
            nonP300Ind = nonP300Ind + 1;
        end
    end
end
tmp = mean(sigP3);
for i = 1 : 7
    sigP3Avrg = sigP3Avrg +  tmp(1,1:160,i) + tmp(1,161:320,i);
end
sigP3Avrg = sigP3Avrg / 14;
tmp = mean(sigNonP3);
for i = 1 : 7
    sigNonP3Avrg = sigNonP3Avrg +  tmp(1,1:160,i) + tmp(1,161:320,i) + tmp(1,321:480,i) + tmp(1,481:640,i) + tmp(1,641:800,i) + tmp(1,801:960,i) + tmp(1,961:1120,i) + tmp(1,1121:1280,i) + tmp(1,1281:1440,i) + tmp(1,1441:1600,i);
end
sigNonP3Avrg = sigNonP3Avrg / 70;
% sigNonP3Avrg = mean(sigNonP3);
% sigTrain = [sigP3, sigNonP3];
% 
%% Downsampling Train and Test Data
sigP3DS = zeros(60,32,7);
sigNonP3DS = zeros(60,160,7);
featureTrain1 = zeros (120,160);
featureTrain2 = zeros (60,12);
for i = 1 : 60
sigP3DS(i,:,:) = downsample(sigP3(i,:,:), 10);
sigNonP3DS(i,:,:) = downsample(sigNonP3(i,:,:), 10);
end
tmp = [sigP3DS, sigP3DS, sigP3DS, sigP3DS, sigP3DS];
featureTrain1(1:60,:) = tmp(:,:,2);
featureTrain1(61:120,:) = sigNonP3DS(:,:,2);
group1 = [ones(1,60), -ones(1,60)];
for i = 1 : 60
    featureTrain2(i,1) = (sum(abs(sigP3DS(i,5:10,2))))/6;
    featureTrain2(i,2) = (sum(abs(sigP3DS(i,21:26,2))))/6;
    for j = 0 : 9
        featureTrain2(i,j+3) = (sum(abs(sigNonP3DS(i,(16*j+5) : (16*j + 10),2))))/6;
    end
end
group2 = [ones(1,2), -ones(1,10)];
sigTest = avrgSig(61:85,:,:);
sigTestDS = zeros(25,192,7);
for i = 1 : 25
    sigTestDS(i,:,:) = downsample (sigTest(i,:,:), 10);
end
featureTest1 = zeros(12,160);
for i = 0 : 11
    featureTest1 (i+1,:) = [sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2), sigTestDS(10,(16*i + 1):(16*i+16),2)];
end
featureTest2 = zeros(25,12);
for i = 1 : 25
    for j = 0 : 11
        featureTrain2(i,j+1) = (sum(abs(sigTestDS(i,(16*j+5) : (16*j + 10),2))))/6;
    end
end
svmStruct = svmtrain(featureTrain1, group1);
class1 = svmclassify(svmStruct, featureTest1);

sigTrainDS = zeros(60,192,7);
for i = 1 : 25
    tmp = sigTrain(i,:,:);
    tmp2 = downsample (tmp, 10);
    sigTrainDS(i,:,:) = tmp2;
end
trainData = zeros (192,7);
testData = zeros (192,7);
trainData(1:32,:) = sigP3AvrgDS(1,:,:);
trainData(33:192,:) = sigNonP3AvrgDS(1,:,:);
testData(:,:) = sigTestDS(10,:,:);
group = [ones(1,32), -ones(1,160)];
svmStruct = svmtrain(trainData, group);
class = svmclassify(svmStruct, testData);
javab = zeros(1,12);
for i = 1 : 12
    javab(i) = sum(class(16*(i-1)+1:16*i));
end
AMPAvrg = zeros(85,12,7);
for l = 1 : 7
    for i = 1 : 85
        for j = 0 : 11
            tmp = avrgSig(i,160*j+1:160*j+160,l);
            tmp = tmp';
            [M,I] = max(tmp);
            AMPAvrg(i,j+1,l) = M;%Amplitude
        end
    end
end
% [la, lala] = sort(AMPAvrg(10,:,2),'descend');
% lala
% la

            
%% %%%%%%%%%%%%%%%%%%% Features %%%%%%%%%%%%%%%%%%%%%%
%Latency & Amplitude & Area & Peak to Peak
LAT = zeros (85,180,7);
AMP = zeros (85,180,7);
LAR = zeros (85,180,7);
AAMP = zeros (85,180,7);
ALAR = zeros (85,180,7);
PAR = zeros (85,180,7);
NAR = zeros (85,180,7);
PP = zeros (85,180,7);
PPT = zeros (85,180,7);
PPS = zeros (85,180,7);
N1P = zeros (85,180,7);
N1PL = zeros (85,180,7);
P3N4 = zeros (85,180,7);
P3N1 = zeros (85,180,7);
for i = 1 : size(sigTrain,3)
    for j = 1 : size(sigTrain,1)
        for k = 0 : 179
            tmp = sigTrain(j,160*k+1:160*k+160,i);
            tmp = tmp';
            [M,I] = max(tmp);
            [MI,In] = min(tmp);
            LAT(j,k+1,i) = I;%Latency
            AMP(j,k+1,i) = M;%Amplitude
            LAR(j,k+1,i) = LAT(j,k+1,i)/AMP(j,k+1,i);
            AAMP(j,k+1,i) = abs(M);
            ALAR(j,k+1,i) = abs(LAT(j,k+1,i)/AMP(j,k+1,i));
            tmp = sum(sigTrain(j,160*k+1:160*k+160,i));
            tmp2 = sum(abs(sigTrain(j,160*k+1:160*k+160,i)));
            PAR(j,k+1,i) = tmp + tmp2;%Positive Area
            NAR(j,k+1,i) = tmp - tmp2;%Negative Area
            PP(j,k+1,i) = M - MI;%Max Peak to Min Peak
            PPT(j,k+1,i) = I - In;%Peak to Peak Time
            PPS(j,k+1,i) = PP(j,k+1,i)/PPT(j,k+1,i);%Peak to Peak Slope
            tmp = sigTrain(j,160*k+12:160*k+41,i);
            [MI,In] = min(tmp);
            N1P(j,k+1,i) = MI;%N100 Amplitude
            N1PL(j,k+1,i) = In;%N100 Latency
            tmp = sigTrain(j,160*k+44:160*k+120,i);
            [M,I] = max(tmp);
            P3N1(j,k+1,i) = M - abs(MI);%Amplitude Difference of P300 and N100
            tmp2 = sigTrain(j,160*k+76:160*k+120,i);
            [MI,In] = min(tmp2);
            P3N4(j,k+1,i) = M - abs(MI);%Amplitude Difference of P300 and N400
            
        end
    end
end
scr1 = 0;
for i = 1 : 85
    a = PAR(i,:,1);
    a = a';
    [la,lala]=sort(a, 'descend');
    if stmType(i,lala(1)) == 1 && stmType(i,lala(2)) == 1
        scr1 = scr1 + 1;
    end
end
scr1
scr2 = 0;
for i = 1 : 85
    a = AMP(i,:,1);
    a = a';
    [la,lala]=sort(a, 'descend');
    if stmType(i,lala(1)) == 1 && stmType(i,lala(2)) == 1
        scr2 = scr2 + 1;
    end
end
scr2
scr3 = 0;
for i = 1 : 85
    a = PPS(i,:,1);
    a = a';
    [la,lala]=sort(a, 'descend');
    if stmType(i,lala(1)) == 1 && stmType(i,lala(2)) == 1
        scr3 = scr3 + 1;
    end
end
scr3
scr4 = 0;
for i = 1 : 85
    a = P3N1(i,:,1);
    a = a';
    [la,lala]=sort(a, 'descend');
    if stmType(i,lala(1)) == 1 && stmType(i,lala(2)) == 1
        scr4 = scr4 + 1;
    end
end
scr4
scr5 = 0;
for i = 1 : 85
    a = P3N4(i,:,1);
    a = a';
    [la,lala]=sort(a, 'descend');
    if stmType(i,lala(1)) == 1 && stmType(i,lala(2)) == 1
        scr5 = scr5 + 1;
    end
end
scr5
scr6 = 0;
for i = 1 : 85
    a = AMPAvrg(i,:,1);
    a = a';
    [la,lala]=sort(a, 'descend');
    if stmTypeAvrg(i,lala(1)) == 1 && stmTypeAvrg(i,lala(2)) == 1
        scr6 = scr6 + 1;
    end
end
scr6
%%%%%%%%%%%%%%%%%%%%%P300 & Non-P300 Epochs Seperation%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------SIGNAL----------------------%
sigNP = zeros(85,24000,7);
sigP = zeros(85,4800,7);
for i = 1 : size(sigTrain,1)
    indxNP = 0;
    indxP = 0;
    for j = 0 : 179
        if stmType(i,j+1) == 0
            sigNP(i,160*indxNP+1:160*indxNP+160,:) = sigTrain(i,42*j+1:42*j+160,:);
            indxNP = indxNP + 1;
        else
            sigP(i,160*indxP+1:160*indxP+160,:) = sigTrain(i,42*j+1:42*j+160,:);
            indxP = indxP + 1;
        end
    end
end
a = sigP(40,:,1);
a = a';
b = sigNP(40,:,1);
b = b';
mean(a)
mean(b)
%------------------FEATURES----------------------%
%CLASS1 = P300 // CLASS2 = No P300
LAT1 = zeros (85,30,7);
LAT2 = zeros (85,150,7);
AMP1 = zeros (85,30,7);
AMP2 = zeros (85,150,7);
LAR1 = zeros (85,30,7);
LAR2 = zeros (85,150,7);
AAMP1 = zeros (85,30,7);
AAMP2 = zeros (85,150,7);
ALAR1 = zeros (85,30,7);
ALAR2 = zeros (85,150,7);
PAR1 = zeros (85,30,7);
PAR2 = zeros (85,150,7);
NAR1 = zeros (85,30,7);
NAR2 = zeros (85,150,7);
PP1 = zeros (85,30,7);
PP2 = zeros (85,150,7);
PPT1 = zeros (85,30,7);
PPT2 = zeros (85,150,7);
PPS1 = zeros (85,30,7);
PPS2 = zeros (85,150,7);
N1P1 = zeros (85,30,7);
N1P2 = zeros (85,150,7);
N1PL1 = zeros (85,30,7);
N1PL2 = zeros (85,150,7);
P3N41 = zeros (85,30,7);
P3N42 = zeros (85,150,7);
P3N11 = zeros (85,30,7);
P3N12 = zeros (85,150,7);
for i = 1 : size(sigTrain,1)
    indxNP = 1;
    indxP = 1;
    for j = 1 : 180
        if stmType(i,j) == 0
            LAT2(i,indxNP,:) = LAT(i,j,:);
            AMP2(i,indxNP,:) = AMP(i,j,:);
            LAR2(i,indxNP,:) = LAR(i,j,:);
            AAMP2(i,indxNP,:) = AAMP(i,j,:);
            ALAR2(i,indxNP,:) = ALAR(i,j,:);
            PAR2(i,indxNP,:) = PAR(i,j,:);
            NAR2(i,indxNP,:) = NAR(i,j,:);
            PP2(i,indxNP,:) = PP(i,j,:);
            PPT2(i,indxNP,:) = PPT(i,j,:);
            PPS2(i,indxNP,:) = PPS(i,j,:);
            N1P2(i,indxNP,:) = N1P(i,j,:);
            N1PL2(i,indxNP,:) = N1PL(i,j,:);
            P3N42(i,indxNP,:) = P3N4(i,j,:);
            P3N12(i,indxNP,:) = P3N1(i,j,:);
            indxNP = indxNP + 1;
        else
            LAT1(i,indxP,:) = LAT(i,j,:);
            AMP1(i,indxP,:) = AMP(i,j,:);
            LAR1(i,indxP,:) = LAR(i,j,:);
            AAMP1(i,indxP,:) = AAMP(i,j,:);
            ALAR1(i,indxP,:) = ALAR(i,j,:);
            PAR1(i,indxP,:) = PAR(i,j,:);
            NAR1(i,indxP,:) = NAR(i,j,:);
            PP1(i,indxP,:) = PP(i,j,:);
            PPT1(i,indxP,:) = PPT(i,j,:);
            PPS1(i,indxP,:) = PPS(i,j,:);
            N1P1(i,indxP,:) = N1P(i,j,:);
            N1PL1(i,indxP,:) = N1PL(i,j,:);
            P3N41(i,indxP,:) = P3N4(i,j,:);
            P3N11(i,indxP,:) = P3N1(i,j,:);
            indxP = indxP + 1;
        end
    end
end

% featureTrain = [LAT1 LAT2;AMP1 AMP2;LAR1 LAR2;AAMP1 AAMP2;ALAR1 ALAR2;PAR1 PAR2;NAR1 NAR2;PP1 PP2;PPT1 PPT2;PPS1 PPS2;N1P1 N1P2;N1PL1 N1PL2;P3N41 P3N42;P3N11 P3N12];
% group = [ones(1,30) -ones(1,150)];
featureTrain = zeros(360,7);
featureTrain(1:30,:) = LAR1(10,:,:);
featureTrain(31:60,:) = N1P1(10,:,:);
featureTrain(61:90,:) = P3N41(10,:,:);
featureTrain(91:120,:) = AAMP1(10,:,:);
featureTrain(121:150,:) = PP1(10,:,:);
featureTrain(151:180,:) = P3N11(10,:,:);

featureTrain(181:210,:) = LAR2(10,61:90,:);
featureTrain(211:240,:) = N1P2(10,61:90,:);
featureTrain(241:270,:) = P3N42(10,61:90,:);
featureTrain(271:300,:) = AAMP2(10,61:90,:);
featureTrain(301:330,:) = PP2(10,61:90,:);
featureTrain(331:360,:) = P3N12(10,61:90,:);

group = [ones(1,180) -ones(1,180)];

% Loading P300 Test Data
[pathstr,~,~] = fileparts(which(mfilename));
d1=[pathstr,'\dataset\Subject_A_Test.mat'];
load(d1);

% convert to double precision
Signal=double(Signal);
Flashing=double(Flashing);
StimulusCode=double(StimulusCode);
target = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU';

tmp = Signal(:,:,chList);
sigTest = zeros(100,28800,7);
for i = 1 : size(sigTest,3)
    for j = 1 : size(sigTest,1)
        for k = 0 : 179
            sigTest(j,160*k+1:160*(k+1),i) = tmp(j,42*k+1:42*k+160,i);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%% Features %%%%%%%%%%%%%%%%%%%%%%
%Latency & Amplitude & Area & Peak to Peak
LATtest = zeros (100,180,7);
AMPtest = zeros (100,180,7);
LARtest = zeros (100,180,7);
AAMPtest = zeros (100,180,7);
ALARtest = zeros (100,180,7);
PARtest = zeros (100,180,7);
NARtest = zeros (100,180,7);
PPtest = zeros (100,180,7);
PPTtest = zeros (100,180,7);
PPStest = zeros (100,180,7);
N1Ptest = zeros (100,180,7);
N1PLtest = zeros (100,180,7);
P3N4test = zeros (100,180,7);
P3N1test = zeros (100,180,7);
for i = 1 : size(sigTest,3)
    for j = 1 : size(sigTest,1)
        for k = 0 : 179
            tmp = sigTest(j,42*k+1:42*k+160,i);
            tmp = tmp';
            [M,I] = max(tmp);
            [MI,In] = min(tmp);
            LATtest(j,k+1,i) = I;%Latency
            AMPtest(j,k+1,i) = M;%Amplitude
            LARtest(j,k+1,i) = LATtest(j,k+1,i)/AMPtest(j,k+1,i);
            AAMPtest(j,k+1,i) = abs(M);
            ALARtest(j,k+1,i) = abs(LATtest(j,k+1,i)/AMPtest(j,k+1,i));
            tmp = sum(sigTest(j,42*k+1:42*k+160,i));
            tmp2 = sum(abs(sigTest(j,42*k+1:42*k+160,i)));
            PARtest(j,k+1,i) = tmp + tmp2;%Positive Area
            NARtest(j,k+1,i) = tmp - tmp2;%Negative Area
            PPtest(j,k+1,i) = M - MI;%Max Peak to Min Peak
            PPTtest(j,k+1,i) = I - In;%Peak to Peak Time
            PPStest(j,k+1,i) = PPtest(j,k+1,i)/PPTtest(j,k+1,i);%Peak to Peak Slope
            tmp = sigTest(j,42*k+12:42*k+41,i);
            [MI,In] = min(tmp);
            N1Ptest(j,k+1,i) = MI;%N100 Amplitude
            N1PLtest(j,k+1,i) = In;%N100 Latency
            tmp = sigTest(j,42*k+44:42*k+120,i);
            [M,I] = max(tmp);
            P3N1test(j,k+1,i) = M - abs(MI);%Amplitude Difference of P300 and N100
            tmp2 = sigTest(j,42*k+76:42*k+120,i);
            [MI,In] = min(tmp2);
            P3N4test(j,k+1,i) = M - abs(MI);%Amplitude Difference of P300 and N400
            
        end
    end
end

%featureTest = [LATtest; AMPtest; LARtest; AAMPtest; ALARtest; PARtest; NARtest; PPtest; PPTtest; PPStest; N1Ptest; N1PLtest; P3N4test; P3N1test];
featureTest = zeros(1080,7);
featureTest(1:180,:) = LARtest(10,:,:);
featureTest(181:360,:) = N1Ptest(10,:,:);
featureTest(361:540,:) = P3N4test(10,:,:);
featureTest(541:720,:) = AAMPtest(10,:,:);
featureTest(721:900,:) = PPtest(10,:,:);
featureTest(901:1080,:) = P3N1test(10,:,:);
a = zeros(180,7);
b = zeros(180,7);
for i = 1 : 180
    a(i,:) = featureTrain(598,i,:);
end
for i = 1 : 180
    b(i,:) = featureTest(703,i,:);
end
class1 = zeros(180,1);
class2 = zeros(180,1);
class3 = zeros(180,1);

Class1 = knnclassify(featureTest, featureTrain, group,5);
SVMStruct = svmtrain(featureTrain,group);
Class2 = svmclassify(SVMStruct, featureTest);
Class3 = classify(featureTest, featureTrain, group);
for i = 1 : 180
    class1(i) = Class1(i) + Class1(i+180) + Class1(i+360) + Class1(i+540) + Class1(i+720) + Class1(i+900);
    class2(i) = Class2(i) + Class2(i+180) + Class2(i+360) + Class2(i+540) + Class2(i+720) + Class2(i+900);
    class3(i) = Class3(i) + Class3(i+180) + Class3(i+360) + Class3(i+540) + Class3(i+720) + Class3(i+900);
    if class1(i) >= 4
        class1(i) = 1;
    elseif class1(i) <= -4
        class1(i) = -1;
    else
        class1(i) = 0;
    end
    if class2(i) >= 4
        class2(i) = 1;
    elseif class2(i) <= -4
        class2(i) = -1;
    else
        class2(i) = 0;
    end
    if class3(i) >= 4
        class3(i) = 1;
    elseif class3(i) <= -4
        class3(i) = -1;
    else
        class3(i) = 0;
    end
end
class = zeros(180,1);
for i = 1 : 180
    class(i) = class1(i) + class2(i) + class3(i);
    if class(i) > -1
        class(i) = 1;
    else
        class(i) = -1;
    end
end

ll = 0;
for i = 1 : 180
    if Class2(i) == 1
        ll = ll + 1;
    end
end
ll
% %%%%%%%%%%%%%%%%%%%%%%%%% Character Guessing %%%%%%%%%%%%%%%%%%%%%%
% 
stmCode = zeros (100, 180);
for i = 1 : 100
    for j = 0 : 179
        stmCode(i,j+1) = StimulusCode(i,42*j + 1);
    end
end

score = zeros(1,12);
for i = 1 : size(class,1)
    if class(i) == 1
        score(stmCode(i)) = score(stmCode(i))+1;
    end
end
[srtScr,indx]=sort(score,'descend');
srtScr
indx

% Train: 60 epochs // Test: 25 epochs
LAT1train = LAT1 (1:60,:,:);
LAT1test = LAT1 (61:85,:,:);
LAT2train = LAT2 (1:60,:,:);
LAT2test = LAT2 (61:85,:,:);
AMP1train = AMP1 (1:60,:,:);
AMP1test = AMP1 (61:85,:,:);
AMP2train = AMP2 (1:60,:,:);
AMP2test = AMP2 (61:85,:,:);
LAR1train = LAR1 (1:60,:,:);
LAR1test = LAR1 (61:85,:,:);
LAR2train = LAR2 (1:60,:,:);
LAR2test = LAR2 (61:85,:,:);
AAMP1train = AAMP1 (1:60,:,:);
AAMP1test = AAMP1 (61:85,:,:);
AAMP2train = AAMP2 (1:60,:,:);
AAMP2test = AAMP2 (61:85,:,:);
ALAR1train = ALAR1 (1:60,:,:);
ALAR1test = ALAR1 (61:85,:,:);
ALAR2train = ALAR2 (1:60,:,:);
ALAR2test = ALAR2 (61:85,:,:);
PAR1train = PAR1 (1:60,:,:);
PAR1test = PAR1 (61:85,:,:);
PAR2train = PAR2 (1:60,:,:);
PAR2test = PAR2 (61:85,:,:);
NAR1train = NAR1 (1:60,:,:);
NAR1test = NAR1 (61:85,:,:);
NAR2train = NAR2 (1:60,:,:);
NAR2test = NAR2 (61:85,:,:);
PP1train = PP1 (1:60,:,:);
PP1test = PP1 (61:85,:,:);
PP2train = PP2 (1:60,:,:);
PP2test = PP2 (61:85,:,:);
PPT1train = PPT1 (1:60,:,:);
PPT1test = PPT1 (61:85,:,:);
PPT2train = PPT2 (1:60,:,:);
PPT2test = PPT2 (61:85,:,:);
PPS1train = PPS1 (1:60,:,:);
PPS1test = PPS1 (61:85,:,:);
PPS2train = PPS2 (1:60,:,:);
PPS2test = PPS2 (61:85,:,:);
N1P1train = N1P1 (1:60,:,:);
N1P1test = N1P1 (61:85,:,:);
N1P2train = N1P2 (1:60,:,:);
N1P2test = N1P2 (61:85,:,:);
N1PL1train = N1PL1 (1:60,:,:);
N1PL1test = N1PL1 (61:85,:,:);
N1PL2train = N1PL2 (1:60,:,:);
N1PL2test = N1PL2 (61:85,:,:);
P3N41train = P3N41 (1:60,:,:);
P3N41test = P3N41 (61:85,:,:);
P3N42train = P3N42 (1:60,:,:);
P3N42test = P3N42 (61:85,:,:);
P3N11train = P3N11 (1:60,:,:);
P3N11test = P3N11 (61:85,:,:);
P3N12train = P3N12 (1:60,:,:);
P3N12test = P3N12 (61:85,:,:);

%Feature1 : Serial Features of P300
%Feature2 : Serial Features of Non-P300

feature1Train = [LAT1train AMP1train LAR1train AAMP1train ALAR1train PAR1train NAR1train PP1train PPT1train PPS1train N1P1train N1PL1train P3N41train P3N11train];
feature1Test = [LAT1test AMP1test LAR1test AAMP1test ALAR1test PAR1test NAR1test PP1test PPT1test PPS1test N1P1test N1PL1test P3N41test P3N11test];

feature2Train = [LAT2train AMP2train LAR2train AAMP2train ALAR2train PAR2train NAR2train PP2train PPT2train PPS2train N1P2train N1PL2train P3N42train P3N12train];
feature2Test = [LAT2test AMP2test LAR2test AAMP2test ALAR2test PAR2test NAR2test PP2test PPT2test PPS2test N1P2test N1PL2test P3N42test P3N12test];

featureTrain = [feature1Train feature2Train];
featureTest = [feature1Test feature2Test];
N0 = 0; N1 = 0;
m0 = zeros(size(sig,1),size(chList,2)); m1 = zeros(size(sig,1),size(chList,2));

for i = 1 : size(sig,3)
    for j = 1 : size(sig,1)
        for k = 1 : size(sig,2)
            if StimulusType(j,k) == 1
                N0 = N0 + 1;
                m0(j,i) = m0(j,i) + sig(j,k,i);
            else
                N1 = N1 + 1;
                m1(j,i) = m1(j,i) + sig(j,k,i);
            end
        end
        m0(j,i) = m0(j,i)/N0;
        m1(j,i) = m1(j,i)/N1;
        N0 = 0;
        N1 = 0;
    end
end

    
%----------------------------------------------------------------------
signal = zeros(180,160);
group = zeros(1,80);
group2 = zeros(1,100);
for i = 0:179
    signal(i+1,:) = Signal(m,42*i+1:42*i+160,11);
end
[B,A] = cheby1(4,.2,[0.1 10]/120);
sigFiltered = filter(B,A,signal);
sigFiltered = signal;
sigTrain = sigFiltered(1:80,:);
sigTest = sigFiltered(81:180,:);
for i = 1:80
    if StimulusType(m,42*(i-1)+1)==1
        group(1,i) = 1;
    else
        group(1,i) = 0;
    end
end
for i = 81:180
    if StimulusType(m,42*(i-1)+1)==1
        group2(1,i-80) = 1;
    else
        group2(1,i-80) = 0;
    end
end
group3 = stepwisefit(sigTrain,group');
class3 = stepwisefit(sigTest', group3);
class3 = class3';
class1 = knnclassify(sigTest, sigTrain, group,3);

wLDA = LDA(sigTrain, group');
w = wLDA(1,1:160) + wLDA(2, 1:160);
class4 = sigTest * w';

ra1 = 0;
for i = 1:100
    if group2(i) == class1(i)
        ra1 = ra1+1;
    end
end
ra1

SVMStruct = svmtrain(sigTrain,group);
class2 = svmclassify(SVMStruct, sigTest);
class2 = class2';

ra2 = 0;
for i = 1:100
    if group2(i) == class2(i)
        ra2 = ra2+1;
    end
end
ra2

ra3 = 0;
for i = 1:100
    if group2(i) == class3(i)
        ra3 = ra3+1;
    end
end
ra3

ra4 = 0;
for i = 1:100
    if group2(i) == class4(i)
        ra4 = ra4+1;
    end
end
ra4

matrix = ['A' 'B' 'C' 'D' 'E' 'F'; 'G' 'H' 'I' 'J' 'K' 'L'; 'M' 'N' 'O' 'P' 'Q' 'R'; 'S' 'T' 'U' 'V' 'W' 'X'; 'Y' 'Z' '1' '2' '3' '4'; '5' '6' '7' '8' '9' '_'];
score = zeros(1,12);
code = zeros(1,100);
for i = 81:180
    code(i-80) = StimulusCode(m,42*(i-1)+1);
end;

for i = 1 : size(class2,2)
    if class2(i) == 1
        score(code(i)) = score(code(i))+1;
    end
end
[srtScr,indx]=sort(score,'descend');

if indx(1) <= 6
    row = indx(1);
    for i = 2 : size(indx,2)-1
        if indx(i) > 6
            col = indx(i);
            break;
        end
    end
else
    col = indx(1);
    for i = 2 : size(indx,2)-1
        if indx(i) < 6
            row = indx(i);
            break;
        end
    end
end

predictedChar = matrix(row, col-6)
TargetChar(m)
sigFiltered = decimate(sigFiltered, 24);
s = Signal(2,:,11);
s1 = [];
s2 = [];
for i = 1 : size(StimulusType,2)
    if StimulusType(2,i)==1
        s1 = [s1 s(i)];
        if StimulusType(2,i+1)== 0
            s1 = [s1 s(i+25:i+160)];
        end
     else
        s2 = [s2 s(i)];
    end
end
s2(size(s1,2)+1:end)=[];
plot(s1)
title('original p300')
figure
plot(s2)
title('original non-p300')
size(s1)
size(s2)
[c1,l1] = wavedec(s1,5,'db1');
[cd11, cd12, cd13] = detcoef(c1,l1,[1 2 3]);
[c2,l2] = wavedec(s2,5,'db1');
[cd21, cd22, cd23] = detcoef(c2,l2,[1 2 3]);
figure
plot(cd13)
hold on
plot(cd23, 'g')
title('level 3')
figure
plot(cd12)
hold on
plot(cd22, 'g')
title('level 2')
figure
plot(cd11)
hold on
plot(cd21 , 'g')
title('level 1')
varp300 = [var(cd11) var(cd12) var(cd13)];
varnonp300 = [var(cd21) var(cd22) var(cd23)];
meanp300 = [mean(cd11) mean(cd12) mean(cd13)];
meannonp300 = [mean(cd21) mean(cd22) mean(cd23)];