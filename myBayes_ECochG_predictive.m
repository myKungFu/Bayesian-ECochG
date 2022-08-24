function [dmin,Sem,Sigma] = myBayes_ECochG_predictive(I,J,N,rm,d,dSigma,type,doPlot)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [dmin,Sem,Sigma] = myBayes_ECochG_predictive(I,J,N,rm,d,dSigma,type,doPlot);
%
% Calculate predictive values for the paper entitled,
% "Minimum Detectable Differences in Electrocochleography Measurements: 
% Bayesian-based Predictions", by Shawn S. Goodman, Jeffery T Lichtenhan, 
% and Skyler Jennings.
%
% This code calculates dmin values from the Bayesian posteriors.
%
% INPUT ARGUMENTS:
% I = number of individuals in the group (1,2,4,8,16, or 32)
% J = number of repeated measurements made on each individual (1,2,4, or 8)
% N = number of simulations to run
% rm = experiment type (1=repeated measures; 0=independent measures)
% d = estimated dmin value (ballpark value); this should be a 1x3 vector
%       for [AP, SP, and SP/AP] estimates.
% dSigma = standard deviation of effect
% type = 'post'
% doPlot = turn on and off plotting (1=plot; 0=don't plot)
%
% REQUIRED FUNTIONS and FILES:
% Data saved by myBayes_ECochG_engine.m. Default names are:
%  SJ_bayesAnalysisGamma_1.mat
%  SJ_bayesAnalysisEpsilon_1.mat
%
% OUTPUT ARGUMENTS:
% dmin = vector of minimum detectable difference (dB) for AP, SP, SP/AP
% Sem = standard error of the mean of dmin (dB) for AP, SP, SP/AP
% Sigma = standard deviation of mean group difference (dB); appendix of paper
% 
% Example Usage:
%   [dmin,Sem,Sigma] = myBayes_ECochG_predictive(4,2,5000,1,[2,3,2],1,'post',1);
%
% Author: Shawn Goodman
% Date: May 14 - August 24, 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if doPlot == 1
        Title1 = 'AP'; % compount action potential
        Title2 = 'SP'; % summating potential
        Title3 = 'APn'; % AP / SP
        Title4 = 'SPn'; % SP / AP
    else
        Title1 = [];
        Title2 = [];
        Title3 = [];
        Title4 = [];
    end

    pathName = 'C:\myWork\ARLas\Peripheral\analysis\SJ\'; % location of saved analysis files
    gFileName = 'SJ_bayesAnalysisGamma_1.mat'; % name of saved analysis file
    eFileName = 'SJ_bayesAnalysisEpsilon_1.mat'; % name of saved analysis file
    [gAP_mu,gAP_sigma,gSP_mu,gSP_sigma,gC_rho,eAP_mu,eAP_sigma,eAP_nu,eSP_mu,...
        eSP_sigma,eSP_nu,eC_rho] = getDistributions(type,pathName,gFileName,eFileName); % get saved posterior distributions

    AgammaBar_AP = zeros(N,1); % initialize 
    AgammaBar_SP = zeros(N,1);
    AgammaBar_APn = zeros(N,1);
    AgammaBar_SPn = zeros(N,1);
    B0gammaBar_AP = zeros(N,1);
    B0gammaBar_SP = zeros(N,1);
    B0gammaBar_APn = zeros(N,1);
    B0gammaBar_SPn = zeros(N,1);
    B1gammaBar_AP = zeros(N,1);
    B1gammaBar_SP = zeros(N,1);
    B1gammaBar_APn = zeros(N,1);
    B1gammaBar_SPn = zeros(N,1);
    for n=1:N % loop across desired number of iterations
        if mod(n,1000)==0
            disp(['      Iteration ',num2str(n),' of ',num2str(N)])
        end

        % from the hyperparameters, draw values for this iteration
        gAPmu = gAP_mu.random;
        gAPsigma = gAP_sigma.random;
        gSPmu = gSP_mu.random;
        gSPsigma = gSP_sigma.random;
        gCrho = gC_rho.random;
        if gCrho <-1
            gCrho = -1;
        end
        if gCrho > 1
            gCrho = 1;
        end
    
        eAPmu = eAP_mu.random;
        eAPsigma = eAP_sigma.random;
        eAPnu = eAP_nu.random;
        eSPmu = eSP_mu.random;
        eSPsigma = eSP_sigma.random;
        eSPnu = eSP_nu.random;
        eCrho = eC_rho.random;
        if eCrho <-1
            eCrho = -1;
        end
        if eCrho > 1
            eCrho = 1;
        end

        % to examine the effet of zero correlation, can force the
        % correlation between AP and SP to zero here:
        % gCrho = 0;
        % eCrho = 0;

    
        % for the baseline condition ----------------------------------
        % GAMMA -----
        % create likelihood distributions
        gAPLike = truncate(makedist('normal',gAPmu,gAPsigma),gAPmu-3*gAPsigma,gAPmu+3*gAPsigma);
        gSPLike = truncate(makedist('normal',gSPmu,gSPsigma),gSPmu-3*gSPsigma,gSPmu+3*gSPsigma);
        % Generate bivariate normal (Z), make into true distribution (Y) using copula (C)
        Z = mvnrnd([0 0], [1 gCrho; gCrho 1], I);
        C = normcdf(Z);
        gamma = [gAPLike.icdf(C(:,1)) gSPLike.icdf(C(:,2))];
        if rm == 1 % if repeated measures, then use same gamma
            gammaB = gamma;
        else % if independent measures, use new gamma
            Z = mvnrnd([0 0], [1 gCrho; gCrho 1], I);
            C = normcdf(Z);
            gammaB = [gAPLike.icdf(C(:,1)) gSPLike.icdf(C(:,2))];
        end

        % EPSILON -----
        % create likelihood distributions
        eAPLike = truncate(makedist('tlocationscale',eAPmu,eAPsigma,eAPnu),-10,10);
        eSPLike = truncate(makedist('tlocationscale',eSPmu,eSPsigma,eSPnu),-10,10);

        AgammaHat_AP = zeros(I,1);
        AgammaHat_SP = zeros(I,1);
        AgammaHat_APn = zeros(I,1);
        AgammaHat_SPn = zeros(I,1);
        B0gammaHat_AP = zeros(I,1);
        B0gammaHat_SP = zeros(I,1);
        B0gammaHat_APn = zeros(I,1);
        B0gammaHat_SPn = zeros(I,1);
        B1gammaHat_AP = zeros(I,1);
        B1gammaHat_SP = zeros(I,1);
        B1gammaHat_APn = zeros(I,1);
        B1gammaHat_SPn = zeros(I,1);
        
        for ii=1:I % loop across individuals
            % for the baseline condition ------------------------------
            % Generate bivariate normal (Z), make into true distribution (Y) using copula (C)
            Z = mvnrnd([0 0], [1 eCrho; eCrho 1], J);
            C = normcdf(Z);
            epsilon = [eAPLike.icdf(C(:,1)) eSPLike.icdf(C(:,2))];
            [AgammaHat_AP(ii,1),AgammaHat_SP(ii,1),AgammaHat_SPn(ii,1)] = calculus(ii,gamma,epsilon,[0 0 0 0],0);

            % for the null/no-change condition -----------------------
            % Generate bivariate normal (Z), make into true distribution (Y) using copula (C)
            Z = mvnrnd([0 0], [1 eCrho; eCrho 1], J);
            C = normcdf(Z);
            epsilon = [eAPLike.icdf(C(:,1)) eSPLike.icdf(C(:,2))];
            [B0gammaHat_AP(ii,1),B0gammaHat_SP(ii,1),B0gammaHat_SPn(ii,1)] = calculus(ii,gammaB,epsilon,[0 0 0 0],0);

            % for the B1 alternative/change condition -----------------------
            % Generate bivariate normal (Z), make into true distribution (Y) using copula (C)
            [B1gammaHat_AP(ii,1),B1gammaHat_SP(ii,1),B1gammaHat_SPn(ii,1)] = calculus(ii,gammaB,epsilon,d,dSigma);

        end
        % average across individuals
        AgammaBar_AP(n,1) = mean(AgammaHat_AP);
        AgammaBar_SP(n,1) = mean(AgammaHat_SP);
        AgammaBar_APn(n,1) = mean(AgammaHat_APn);
        AgammaBar_SPn(n,1) = mean(AgammaHat_SPn);

        B0gammaBar_AP(n,1) = mean(B0gammaHat_AP);
        B0gammaBar_SP(n,1) = mean(B0gammaHat_SP);
        B0gammaBar_APn(n,1) = mean(B0gammaHat_APn);
        B0gammaBar_SPn(n,1) = mean(B0gammaHat_SPn);
        
        B1gammaBar_AP(n,1) = mean(B1gammaHat_AP);
        B1gammaBar_SP(n,1) = mean(B1gammaHat_SP);
        B1gammaBar_APn(n,1) = mean(B1gammaHat_APn);
        B1gammaBar_SPn(n,1) = mean(B1gammaHat_SPn);
    end

    [deltaMin_AP,sem_AP,sigma_AP] = calculateDelta(AgammaBar_AP,B0gammaBar_AP,B1gammaBar_AP,N,d(1),Title1,dSigma,I);
    [deltaMin_SP,sem_SP,sigma_SP] = calculateDelta(AgammaBar_SP,B0gammaBar_SP,B1gammaBar_SP,N,d(2),Title2,dSigma,I);
    [deltaMin_SPn,sem_SPnorm,sigma_SPnorm] = calculateDelta(AgammaBar_SPn,B0gammaBar_SPn,B1gammaBar_SPn,N,d(3),Title4,dSigma,I);

    dmin = [deltaMin_AP,deltaMin_SP,deltaMin_SPn]; % smallest detectable difference
    Sem = [sem_AP,sem_SP,sem_SPnorm]; % standard error of the mean of dmin
    Sigma = [sigma_AP,sigma_SP,sigma_SPnorm]; % standard deviation of mean group difference

end

% INTERNAL FUNCTIONS ------------------------------------------------------
function [gammaHat_AP,gammaHat_SP,gammaHat_SPn] = calculus(ii,gamma,epsilon,D,dSigma)
    % The calculus of EChochG: the smallest detectable differences 
    AP = gamma(ii,1) + epsilon(:,1); % This is AP alone
    SP = gamma(ii,2) + epsilon(:,2); % This is SP alone
    if dSigma ~= 0 % represent the effect size variance
        pd = makedist('normal',D(1),dSigma);
        d1 = pd.random(1,1);
        pd = makedist('normal',D(2),dSigma);
        d2 = pd.random(1,1);
        pd = makedist('normal',D(3),dSigma);
        d4 = pd.random(1,1);
    else
        d1 = D(1);
        d2 = D(2);
        d4 = D(4);
    end
    gammaHat_AP = mean(AP+d1); % take mean across n repeated measurements
    gammaHat_SP = mean(SP+d2);
    gammaHat_SPn = mean(SP+d4-AP);
end
function [deltaMin,sem,sigma0] = calculateDelta(gammaBar_AP,gammaBar_APb0,gammaBar_APb1,N,d,Title,dSigma,I)
    if ~isempty(Title)
        doPlot = 1;
    else
        doPlot = 0;
    end
    % inital clean up, if necessary
    indx = find(isnan(gammaBar_APb1));
    if ~isempty(indx)
        pcReject = length(indx)/length(gammaBar_APb1);
        gammaBar_AP(indx) = [];
        gammaBar_APb0(indx) = [];
        gammaBar_APb1(indx) = [];
    else
        pcReject = 0;
    end

    % Set Levels -------------------------------------------
    alpha = 0.05;
    beta = 0.2;

    % SHORTCUT METHOD: -------------------------------------
    delta0 = gammaBar_APb0 - gammaBar_AP;
    sigma0 = std(delta0); %standard deviation
    sigma1 = sqrt(sigma0^2+(dSigma.^2/(I)));
    pd0 = makedist('Normal',0,sigma0);
    pd1 = makedist('Normal',0,sigma1);
    if d >= 0
        deltaMin = pd0.icdf(1-alpha)+pd1.icdf(1-beta);
    else
        deltaMin = pd0.icdf(alpha)+pd1.icdf(beta);
    end

    delta1 = gammaBar_APb1 - gammaBar_AP;
    sem = sqrt(var(delta0) + var(delta1))/sqrt(N); % same as above

%     % LONGER WAY: ------------------------------------------
%     delta1 = gammaBar_APb1 - gammaBar_AP;
%     sigma1L = std(delta1); %standard deviation
%     %pd0 = makedist('Normal',0,sigma0);
%     pd1L = makedist('Normal',0,sigma1L);
%     if d >= 0
%         deltaMinL = pd0.icdf(1-alpha)+pd1L.icdf(1-beta);
%     else
%         deltaMinL = pd0.icdf(alpha)+pd1L.icdf(beta);
%     end
%
    if doPlot == 1

    % LONGEST WAY: ----------------------------------------
        pd0 = fitdist(delta0,'normal');
        pd1 = fitdist(delta1,'normal');
        xmin = min([min(delta0),min(delta1)]);
        xmax = max([max(delta0),max(delta1)]);
        xxx = linspace(xmin,xmax,10000);
        if length(xxx) > 10000
            warning('length of xxx too long.')
            keyboard
        end
        D0_cdf = pd0.cdf(xxx);
        D1_cdf = pd1.cdf(xxx);
    
        % This is for a 1-tailed test
        if d >= 0
            [~,indx1] = min(abs(D0_cdf-(1-alpha)));
            cut95 = xxx(indx1);
            [~,indx2] = min(abs(D1_cdf-beta));
            cut80 = xxx(indx2);
        else    
            [~,indx1] = min(abs(D0_cdf-(alpha)));
            cut95 = xxx(indx1);
            [~,indx2] = min(abs(D1_cdf-(1-beta)));
            cut80 = xxx(indx2);
        end    
    
        adjust = cut95-cut80;
        deltaMinLL = d + adjust; % estimated minimum detectable difference
        
        delta0(isnan(delta0)) = [];
        delta1(isnan(delta1)) = [];

        figure
        histogram(delta0,'Normalization','pdf')
        hold on
        histogram(delta1,'Normalization','pdf')
        plot(xxx,pd0.pdf(xxx),'b')
        plot(xxx,pd1.pdf(xxx),'r')

        maxy = max([max(pd0.pdf(xxx)),max(pd1.pdf(xxx))]);
        maxy = maxy + (0.05*maxy);
        line([cut80,cut80],[0 maxy],'LineStyle','-','Color',[1 0 0])
        line([cut95,cut95],[0 maxy],'LineStyle','--','Color',[0 0 1])

        title([Title,'   ',num2str(pcReject*100,2),'% Rejections'])
    end
end
function [gAP_mu,gAP_sigma,gSP_mu,gSP_sigma,gC_rho,eAP_mu,eAP_sigma,eAP_nu,eSP_mu,eSP_sigma,eSP_nu,eC_rho] = getDistributions(type,pathName,gFileName,eFileName)
    warning off
    dummy = load([pathName,gFileName]);
    gMCAP = dummy.MCMC_AP;
    gMCSP = dummy.MCMC_SP;
    gMCC = dummy.MCMC_C;
    gModAP = dummy.Model_AP;
    gModSP = dummy.Model_SP;
    gModC = dummy.Model_C;
    dummy = load([pathName,eFileName]);
    eMCAP = dummy.MCMC_AP;
    eMCSP = dummy.MCMC_SP;
    eMCC = dummy.MCMC_C;
    eModAP = dummy.Model_AP;
    eModSP = dummy.Model_SP;
    eModC = dummy.Model_C;
    clear dummy
    warning on
    if strcmp(type,'prior') % create the prior distributions --------------
        if size(Model.prior.names,2) == 2
            pd1 = makedist('normal',Model.prior.values{1}(1),Model.prior.values{1}(2));
            pd2 = truncate(makedist('tlocationscale',Model.prior.values{2}(1),Model.prior.values{2}(2),1),0,50);
        elseif size(Model.prior.names,2) == 3
            pd1 = truncate(makedist('normal',Model.prior.values{1}(1),Model.prior.values{1}(2)),-5,5); % mu = normal prior
            pd2 = truncate(makedist('tlocationscale',Model.prior.values{2}(1),Model.prior.values{2}(2),1),0,10); % sigma = half cauchy
            pd3 = truncate(makedist('gamma',Model.prior.values{3}(1),Model.prior.values{3}(2)),1,100); % nu = beta
        end
    elseif strcmp(type,'post') % create the posterior distribution hyperparameters -------
        gAP_mu = gModAP.Posterior1.dist;
        gAP_sigma = gModAP.Posterior2.dist;
        gSP_mu = gModSP.Posterior1.dist;
        gSP_sigma = gModSP.Posterior2.dist;
        try
            gC_rho = gModC.Posterior1.dist;
        catch
            gC_rho = gModC.Posterior1_C.dist;
        end
        %
        eAP_mu = eModAP.Posterior1.dist;
        eAP_sigma = eModAP.Posterior2.dist;
        eAP_nu = eModAP.Posterior3.dist;
        eSP_mu = eModSP.Posterior1.dist;
        eSP_sigma = eModSP.Posterior2.dist;
        eSP_nu = eModSP.Posterior3.dist;
        try
            eC_rho = eModC.Posterior1.dist;
        catch
            eC_rho = eModC.Posterior1_C.dist;
        end
    end
end
