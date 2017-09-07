%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all ;close all;
addpath(genpath('./'));
%rng('default');					% ???
%load bibtex_test

% load MovieLens 100K dataset
load ../gendata/ml-100k/ml1.mat
% load MovieLens 1M dataset
%load ../gendata/ml-1m/ml1.mat
UIdx = train_label(:,1);								% indices of users in training data
TUIdx = test_label(:,1);								% indices of users in testing data
train_x = feat_rating(UIdx, :);
test_x = feat_rating(TUIdx, :);
train_y = train_label(:, 2:end);						
test_y = test_label(:, 2:end);							% remove uesr ids

st = cputime;
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
    print_opt(opt);
    dbn = dbnsetup(v,[800]);
    dbn = dbntrain(dbn,train_x,opt);
    for a = 1 :1
        for b = 1 : 1
            [ train_x,train_y,test_x,test_y ] = generateCVSet( X,Y,kk,cv, totalCV);
            
            nn = dbntonn(dbn);
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
