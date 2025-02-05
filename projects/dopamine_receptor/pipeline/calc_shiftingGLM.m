function GLMstruct = calc_shiftingGLM(responses,predictors,Fs,varargin)
%predictors - variables same size as response in a matrix that is
%nFrames x nPredictors

%response  - variables that you want to regress out speed/see fit should be
%nFrames x nResponses

%Fs - Framerate of sampling to standardize shifts in time (seconds) to
%frames

%varargin
%1st - shifts if you want to give as a a vector of timings right now only
%supporst positive shifts (AKA predictors shifted backwards in time, AKA
%the response is allowed to lag the predictor)

if nargin>3
    shifts = varargin{1};
else
    shifts = 0:0.25:3;
end

% would be response
%                                     Fall = mouseP(m).detrFalls{d};
% would be predictors
%                                     forwards = mouseP(m).Forwards{d};
%                                     rotation = mouseP(m).Rotations{d};
%                                     yposfam = mouseP(m).ybinned{d}{e}(:,1);
%                                     totfam = 0.49*sqrt(forwards{e}(:,1).^2+rotation{e}(:,1).^2);
%                                     acceltot = [0; diff(smoothdata(totfam,'rlowess',round(2*FsPV(m,d))))]*FsPV(m,d);
%                                     accelfor = [0; diff(smoothdata(forwards{e}(:,1),'rlowess',round(2*FsPV(m,d))))*FsPV(m,d)];
%                                     accelrot = [0; diff(smoothdata(rotation{e}(:,1),'rlowess',round(2*FsPV(m,d))))*FsPV(m,d)];
%                                     VRspeed = [0; diff(yposfam)];
%                                     VRaccel = [0; diff(smoothdata(VRspeed,'rlowess',round(2*FsPV(m,d))))*FsPV(m,d)];
%                                     x = [smoothdata(forwards{e}(:,1),'gaussian',round(2*FsPV(m,d))),smoothdata(rotation{e}(:,1),'gaussian',round(2*FsPV(m,d))),smoothdata(totfam,'gaussian',round(2*FsPV(m,d))),acceltot,abs(acceltot),accelfor,accelrot,abs(accelfor),abs(accelrot),smoothdata(VRspeed,'gaussian',round(2*FsPV(m,d))),VRaccel,abs(VRaccel)];


x = predictors;
for c = 1:size(predictors,2)
    for s = 1:length(shifts)
        y =predictors(:,c);
        modelType = 'linear';
        distribution = 'normal';
        linktype = 'identity';
        modTemp=fitglm(x(1:end-ceil(Fs*shifts(s)),:),y(ceil(Fs*shifts(s))+1:end),modelType,'Distribution',distribution,'Link',linktype);
        fcn=@(Xtr,Ytr,Xte) predict(fitglm(Xtr,Ytr,modelType,...
            'Distribution',distribution,'Link',linktype),Xte);
        fcnCorr=@(Xtr,Ytr,Xte,Yte) corrcoef(predict(fitglm(Xtr,Ytr,modelType,...
            'Distribution',distribution,'Link',linktype),Xte),Yte);

        mse=modTemp.SSE/length(y(ceil(Fs*shifts(s)+1):end));
        msecv=crossval('mse',x(1:end-ceil(Fs*shifts(s)),:),y(ceil(Fs*shifts(s))+1:end),'Predfun',fcn,'kfold',10);
        GLMstruct.rmse{s}(c)= sqrt(msecv);
        corrcvT=crossval(fcnCorr,x(1:end-ceil(Fs*shifts(s)),:),y(ceil(Fs*shifts(s))+1:end),'kfold',10);
        corrcvT=mean(corrcvT);
        corrcvT=corrcvT(2);
        GLMstruct.rmseALL{s}(c) = sqrt(mse);
        % %                                             %                    fitted{k,j}=modTemp.Fitted{:,1};
        corrT=corrcoef(modTemp.Fitted{:,1},y(ceil(Fs*shifts(s))+1:end));
        GLMstruct.Corrs{s}(c) = corrT(1,2);
        GLMstruct.CorrsALL{s}(c) = corrcvT;
        GLMstruct.AllModelTraces{s}(:,c) = [NaN(ceil(Fs*shifts(s)),1); modTemp.Fitted{:,1}];
        GLMstruct.AllModelFalls{s}(:,c) = y - [NaN(ceil(Fs*shifts(s)),1); modTemp.Fitted{:,1}];
        GLMstruct.AllGLMModels{s}{c} = modTemp;
    end
    temp = cell2mat(GLMstruct.rmse');
    temp2 = cell2mat(GLMstruct.Corrs');

    [~,minerror] = min(temp(:,size(temp,2)));
    [~,maxcorr] = max(temp2(:,size(temp2,2)));
    GLMstruct.minerrorshift(c) = minerror;
    GLMstruct.maxcorrshift(c) = maxcorr;
%     GLMstruct(:,c) = GLMstruct.minerror(:,c);
%     GLMstruct(:,c) = GLMstruct.maxcorr(:,c);
%     GLMstruct(:,c) = GLMstruct.minerror(:,c);
%     GLMstruct(:,c) = GLMstruct.maxcorr(:,c);
end
end