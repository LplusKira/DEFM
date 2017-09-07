%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all ;close all;
addpath(genpath('./'));
%rng(0);

%rng('default');		% ???

%load bibtex_test

% load MovieLens 100K generated dataset
%load ../gendata/ml-100k/ml1.mat
% load MovieLens 1M generated dataset
load ../gendata/ml-1m/ml1.mat
fprintf('factorize sparse features by bayespmf.\n');
k = 100;
mf_iter = 50;
tr_rate = 0.8;		% the rate of training data to feed BPMF
restart = 1;
%tr_m = size(train_feat, 1);
%te_m = size(test_feat, 1);
[uidxs, iidxs, values] = find(feat_rating);
all_triplets = [uidxs, iidxs, values];
all_triplets = all_triplets(randperm(length(all_triplets)), :);															% randomly shuffle triplets
all_tr_triplets = all_triplets(1:int32(length(all_triplets)*tr_rate), :);
all_val_triplets = all_triplets(int32(length(all_triplets)*tr_rate)+1:length(all_triplets), :);							% split data into train, valid to BPMF
%[train_uidxs, train_iidxs, train_values] = find(train_feat);
%[test_uidxs, test_iidxs, test_values] = find(test_feat);
%train_triplets = [train_uidxs, train_iidxs, train_values];
%test_triplets = [test_uidxs, test_iidxs, test_values];
%train_triplets = train_triplets(randperm(length(train_triplets)), :);					
%test_triplets = test_triplets(randperm(length(test_triplets)), :);														% randomly shuffle triplets
%train_tr_triplets = train_triplets(1:int32(length(train_triplets)*tr_rate), :);
%train_val_triplets = train_triplets(int32(length(train_triplets)*tr_rate)+1:length(train_triplets), :);	
%test_tr_triplets = test_triplets(1:int32(length(test_triplets)*tr_rate), :);
%test_val_triplets = test_triplets(int32(length(test_triplets)*tr_rate)+1:length(test_triplets), :);					
%all_tr_triplets = [train_tr_triplets; test_tr_triplets];
%all_val_triplets = [train_val_triplets; test_val_triplets];															% split data into train, valid to BPMF

st = cputime;
%[tr_w1_U1, tr_w1_V1] = pmf(restart, k, train_tr_triplets, train_val_triplets);
%[tr_w1_U1_smp, tr_w1_V1_smp] = bayespmf(restart, k, tr_w1_U1, tr_w1_V1, train_tr_triplets, train_val_triplets);
%[te_w1_U1, te_w1_V1] = pmf(restart, k, test_tr_triplets, test_val_triplets);
%[te_w1_U1_smp, te_w1_V1_smp] = bayespmf(restart, k, te_w1_U1, te_w1_V1, test_tr_triplets, test_val_triplets);		
[w1_U1, w1_V1] = pmf(restart, k, mf_iter, all_tr_triplets, all_val_triplets);
[w1_U1_smp, w1_V1_smp] = bayespmf(restart, k, mf_iter, w1_U1, w1_V1, all_tr_triplets, all_val_triplets);						% run BPMF
disp(size(w1_U1_smp));
disp(size(w1_V1_smp));

UIdx = train_label(:,1);																							% indices of users in training data
TUIdx = test_label(:,1);
train_x = w1_U1_smp(UIdx, :);
test_x = w1_U1_smp(TUIdx, :);
train_y = train_label(:, 2:end);
test_y = test_label(:, 2:end);																						% remove user ids
X = [train_x;test_x];
Y = [train_y;test_y];
X = double(X); Y = double(Y);
v = size(train_x,2);
y = size(train_y,2);
totalCV = 10;
kk = randperm(size(X,1));
RL = []; COV = []; PRE = []; ONE = []; HL = []; MI = []; MA= [];
%LR = [0.05,0.005,0.0005];
%WD = [0.05,0.005,0.0005];1
for cv = 1 : totalCV
    opt.lr = 0.05;
    opt.momen = 0.5;
    opt.wd = 0.00005;
    opt.rho = 0.05;
    opt.beta = 0.0005;
    opt.b_s = ceil(size(train_x,1) / 50);
    opt.max_iter =100;
    opt.max_cd = 5;
    opt.GAUSSIAN = 0;
    fprintf('Learn a deep belief network to initialize the weights in Auto-Encoder.\n');
    %disp(size(train_triplets));
    %disp(size(test_triplets));
    %disp(size(tr_w1_U1));
    %disp(size(tr_w1_V1));
    %disp(size(tr_w1_U1_smp));
    %disp(size(tr_w1_V1_smp));
    %disp(size(te_w1_U1));
    %disp(size(te_w1_V1));
    %disp(size(te_w1_U1_smp));
    %disp(size(te_w1_V1_smp));
    print_opt(opt);
    
    %dbn = dbnsetup(v,[800]);
    %dbn = dbntrain(dbn,train_x,opt);
    
    for a = 1 :1
        for b = 1 : 1
            [ train_x,train_y,test_x,test_y ] = generateCVSet( X,Y,kk,cv, totalCV);
            
            %nn = dbntonn(dbn);
            nn = nnsetup([k, 800]);
            
            crbm = CRBM_init(nn.size(end) , 600 , y);
            nn = nnfftocrbm(nn,train_x,train_y,train_y,crbm);
            TRX = nn.a{nn.n};
            nn = nnfftocrbm(nn,test_x,test_y,test_y,crbm);
            TEX = nn.a{nn.n};
            opt.wd = 0.05;
            opt.lr = 0.05;
            opt.max_cd = 10;
            fprintf('Learn top-level CRBM using contrastive divergence.\n');
            print_opt(opt);
          %  crbm = CRBM_train(crbm,TRX,train_y,opt);
            opt.lr = 0.005;
            opt.wd = 0.005;
            nn.alpha = opt.lr;nn.momentum = opt.momen;nn.wd=opt.wd;
            fprintf('Fine-Tune the whole model using CSBP.\n');
            [ nn,crbm ] = NCEMjointTrain( nn,crbm,train_x,train_y,opt,test_x,test_y );
            [pred_] = NCEMTestGibbs(nn,crbm,test_x,test_y,2000);
            cv_pred{cv} = pred_;
            rankingloss(pred_,test_y)
        end
    end
end
et = cputime;
fprintf('cost time: %.4f\n', et - st);
[ tau_cv ] = getTauCV( cv_pred , X,Y,kk,totalCV );
for cv = 1 : totalCV
    [ ~,~,test_x,test_y ] = generateCVSet( X,Y,kk,cv, totalCV);
    pred = cv_pred{cv};
    [rl,cov,pre,one,mi_f,hl] = get_all_measure(pred ,  test_y , tau_cv);
    RL = [RL,rl];COV=[COV,cov];PRE=[PRE,pre];ONE=[ONE,one];MI=[MI,mi_f];HL=[HL,hl];
end
