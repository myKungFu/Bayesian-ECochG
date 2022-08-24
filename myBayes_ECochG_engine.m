function [Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C] = myBayes_ECochG_engine(Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Model,MCMC] = myBayes_ECochG_engine(Model,MCMC);
%
% Calculate Bayesian statistics for the paper entitled,
% "Minimum Detectable Differences in Electrocochleography Measurements: 
% Bayesian-based Predictions", by Shawn S. Goodman, Jeffery T Lichtenhan, 
% and Skyler Jennings.
%
% Metropolis-Hastings MCMC sampler, implemented using component-wise sampling.
% Data are standardized for sampling, and the the processed is reversed on
% the sampled coefficients to return them in their original units.
% This engine is to be called by the function myBayes_ECochG.m.
%
% INPUT ARGUMENTS:
% Model = a structure containing all of the needed Bayesian model
%           specifications (see myBayes_ECochG.m)
% MCMC = a structure containing all of the MCMC information.
%           (see myBayes_ECochG.m)
%
% Author: Shawn Goodman
% Date: January 23-August 24, 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Model = Model_AP;
    MCMC = MCMC_AP;

    % Perform some initial housekeeping -----------------------------------
    Model.data.dependent.N = size(Model.data.dependent.y,1); % number of data points in data set
    Model = standardize(Model);  % standardize all the data (both independent and dependent variables)
    MCMC.currentValues = Model.likelihood.startingValues; % initialize starting values for the parameters
    MCMC.proposedValues = Model.likelihood.startingValues * rand(1,1)/2+.5; % initialize values
    MCMC.nSamples = MCMC.nSamples*MCMC.thinning + MCMC.burnInSamples; % get more than requested, due to thinning and burn in
    MCMC.History = zeros(MCMC.nSamples,Model.likelihood.nParameters); % initialize sample history
    MCMC.History(1,:) = MCMC.currentValues; % initialize 
    MCMC.decisionHx = zeros(MCMC.nSamples,Model.likelihood.nParameters); % initialize decision history to zeros (not accept)
    MCMC.counter = 1; % initialize sampler iteration counter
    
    Model_SP.data.dependent.N = size(Model_SP.data.dependent.y,1); % number of data points in data set
    Model_SP = standardize(Model_SP);  % standardize all the data (both independent and dependent variables)
    MCMC_SP.currentValues = Model_SP.likelihood.startingValues; % initialize starting values for the parameters
    MCMC_SP.proposedValues = Model_SP.likelihood.startingValues; % initialize values
    MCMC_SP.nSamples = MCMC_SP.nSamples*MCMC_SP.thinning + MCMC_SP.burnInSamples; % get more than requested, due to thinning and burn in
    MCMC_SP.History = zeros(MCMC_SP.nSamples,Model_SP.likelihood.nParameters); % initialize sample history
    MCMC_SP.History(1,:) = MCMC_SP.currentValues; % initialize 
    MCMC_SP.decisionHx = zeros(MCMC_SP.nSamples,Model_SP.likelihood.nParameters); % initialize decision history to zeros (not accept)
    MCMC_SP.counter = 1; % initialize sampler iteration counter

    Model_C.data.dependent.N = size(Model_C.data.dependent.y,1); % number of data points in data set
    MCMC_C.currentValues = Model_C.likelihood.startingValues; % initialize starting values for the parameters
    MCMC_C.proposedValues = Model_C.likelihood.startingValues; % initialize values
    origSamples = MCMC_C.nSamples; % final number of requested samples
    MCMC_C.nSamples = MCMC_C.nSamples*MCMC_C.thinning + MCMC_C.burnInSamples; % get more than requested, due to thinning and burn in
    MCMC_C.History = zeros(MCMC_C.nSamples,Model_C.likelihood.nParameters); % initialize sample history
    MCMC_C.History(1,:) = MCMC_C.currentValues; % initialize 
    MCMC_C.decisionHx = zeros(MCMC_C.nSamples,Model_C.likelihood.nParameters); % initialize decision history to zeros (not accept)
    MCMC_C.counter = 1; % initialize sampler iteration counter

    reportStep = 100; % report progress to command window this often (number of iterations)
    tuningStep = 1000;  % how often to adjust the tuning parameters (number of iterations)

    % Used for gamma:
    w = 1./Model.likelihood.weighting; % weighting values
    % Used for epsilon:
    starts = Model.likelihood.starts; % vector of indices showing where each participant is located
    finishes = Model.likelihood.finishes; % matrix of amplitudes (SP)
%    m = Model.likelihood.m; % number of repeated measures for each participant
    
    % Run the sampler -----------------------------------------------------
    for nn=1:MCMC.nSamples % ----------------------------------------------
        if mod(nn,reportStep)==0 % report sampling progress
            disp(['   ',num2str(nn),' of ',num2str(MCMC.nSamples),' iterations'])
        end
        if mod(nn,tuningStep)==0 % adjust proposal tuning
            MCMC = adjustTuning(MCMC);
            MCMC_SP = adjustTuning(MCMC_SP);
            MCMC_C = adjustTuning(MCMC_C);
        end
        
        if Model.likelihood.nParameters == 2
            d = Model.data.dependent.z;
            dSP = Model_SP.data.dependent.z;
        elseif Model.likelihood.nParameters == 3
            if ~isempty(starts)
                d = zeros(length(starts),1);
                dSP = d;
                for ii=1:length(starts)
                    q = Model.data.dependent.z(starts(ii):finishes(ii));
                    qSP = Model_SP.data.dependent.z(starts(ii):finishes(ii));
                    d(ii,1) = q(2);
                    dSP(ii,1) = qSP(2);
                end
            else
                d = Model.data.dependent.z;
                dSP = Model_SP.data.dependent.z;
            end
        end

        % Implement component-wise sampling: loop over each parameter
        % All parameters are considered once for each iteration (nn)
        for jj=1:Model.likelihood.nParameters 
            % start with all proposed parameter values equal to the current parameter values
            MCMC.proposedValues = MCMC.currentValues; 
            
            % get new proposal parameter value ---------------------------------
            pd1 = MCMC.proposal.distributions{jj}(MCMC.currentValues(jj),MCMC.proposalTuning(jj)); % create distribution object using current values
            MCMC.proposedValues(1,jj) = pd1.random(1); % randomly draw one sample from the distribution and update the proposed values with it
            % account for assymmetry in the proposal distributions (metropolis-hastings)
            pd2 = MCMC.proposal.distributions{jj}(MCMC.proposedValues(jj),MCMC.proposalTuning(jj)); % create distribution object using proposed values
            C = pd1.pdf(MCMC.currentValues(jj)) / pd2.pdf(MCMC.proposedValues(jj)); % ratio gives a correction value (C) account for assymmetry 
            LC = log(C); % convert to log space (LC) so compatible with log likelihoods and log priors

            % calculate the log-likelihoods -----------------------------------
            % calculate assuming the proposed parameter values
            if Model.likelihood.nParameters == 2
                f = Model.likelihood.equation{1}(d,MCMC.proposedValues(1),MCMC.proposedValues(2)); % likelihood of the data
            elseif Model.likelihood.nParameters == 3
                f = Model.likelihood.equation{1}(d,MCMC.proposedValues(1),MCMC.proposedValues(2),MCMC.proposedValues(3)); % likelihood of the data
            end
            if ~isempty(w) % if weighted log likelihood (LL) is desired
                LL_prop = sum(log(f.^w)); 
            else % otherwise use standard log likelihood (LL)
                LL_prop = sum(log(f)); 
            end
            if Model.likelihood.nParameters == 2 % gamma has two parameters (normal distribution model)
                f = Model.likelihood.equation{1}(d,MCMC.currentValues(1),MCMC.currentValues(2));
            elseif Model.likelihood.nParameters == 3 % epsilon has three parameters (t-location-scale model)
                f = Model.likelihood.equation{1}(d,MCMC.currentValues(1),MCMC.currentValues(2),MCMC.currentValues(3));
            end
            if ~isempty(w) % if weighted likelihood (LL) is desired
                LL_cur = sum(log(f.^w));
            else % otherwise use standard likelihood (LL)
                LL_cur = sum(log(f));
            end

            % get the log of the prior ----------------------------------------
            % calculate assuming the proposed parameter values
            f = Model.prior.equations{jj}(MCMC.proposedValues(jj),Model.prior.values{jj}(1),Model.prior.values{jj}(2));
            PP_prop = log(f); % convert to log prior (PP)
            % calculate assuming the current parameter values
            f = Model.prior.equations{jj}(MCMC.currentValues(jj),Model.prior.values{jj}(1),Model.prior.values{jj}(2));
            PP_cur = log(f);

            % calculate the corrected acceptance ratio -------------------------------------
            WLR = exp((LL_prop+PP_prop)-(LL_cur+PP_cur)+LC); % weighted likelihood ratio
            A = min([1,WLR]); % acceptance criterion
            r = rand(1,1); % random draw from uniform distribution
            if r <= A % if random draw less than acceptance
                MCMC.decisionHx(nn,jj) = 1; % accept proposal
                MCMC.currentValues(jj) = MCMC.proposedValues(jj); % make curent position the proposed
            else
                % decisionHx(1,jj) = 0; % reject proposal
                % this is not used since initialization of matrix is zeros
            end
            MCMC.History(MCMC.counter,jj) = MCMC.currentValues(jj);  % update the sampling history
        end
        MCMC.counter = MCMC.counter +1; % update the iteration counter

     % Now run for SP ------------------
        for jj=1:Model_SP.likelihood.nParameters 
            % start with all proposed parameter values equal to the current parameter values
            MCMC_SP.proposedValues = MCMC_SP.currentValues; 
            
            % get new proposal parameter value ---------------------------------
            pd1 = MCMC_SP.proposal.distributions{jj}(MCMC_SP.currentValues(jj),MCMC_SP.proposalTuning(jj)); % create distribution object using current values
            MCMC_SP.proposedValues(1,jj) = pd1.random(1); % randomly draw one sample from the distribution and update the proposed values with it
            % account for assymmetry in the proposal distributions (metropolis-hastings)
            pd2 = MCMC_SP.proposal.distributions{jj}(MCMC_SP.proposedValues(jj),MCMC_SP.proposalTuning(jj)); % create distribution object using proposed values
            C = pd1.pdf(MCMC_SP.currentValues(jj)) / pd2.pdf(MCMC_SP.proposedValues(jj)); % ratio gives a correction value (C) account for assymmetry 
            LC = log(C); % convert to log space (LC) so compatible with log likelihoods and log priors

            % calculate the log-likelihoods -----------------------------------
            % calculate assuming the proposed parameter values
            if Model_SP.likelihood.nParameters == 2
                f = Model_SP.likelihood.equation{1}(dSP,MCMC_SP.proposedValues(1),MCMC_SP.proposedValues(2)); % likelihood of the data
            elseif Model_SP.likelihood.nParameters == 3
                f = Model_SP.likelihood.equation{1}(dSP,MCMC_SP.proposedValues(1),MCMC_SP.proposedValues(2),MCMC_SP.proposedValues(3)); % likelihood of the data
            end
            if ~isempty(w) % if weighted log likelihood (LL) is desired
                LL_prop = sum(log(f.^w)); 
            else % otherwise use standard log likelihood (LL)
                LL_prop = sum(log(f)); 
            end
            if Model_SP.likelihood.nParameters == 2
                f = Model_SP.likelihood.equation{1}(dSP,MCMC_SP.currentValues(1),MCMC_SP.currentValues(2));
            elseif Model_SP.likelihood.nParameters == 3
                f = Model_SP.likelihood.equation{1}(dSP,MCMC_SP.currentValues(1),MCMC_SP.currentValues(2),MCMC_SP.currentValues(3));
            end
            if ~isempty(w) % if weighted likelihood (LL) is desired
                LL_cur = sum(log(f.^w));
            else % otherwise use standard likelihood (LL)
                LL_cur = sum(log(f));
            end

            % get the log of the prior ----------------------------------------
            % calculate assuming the proposed parameter values
            f = Model_SP.prior.equations{jj}(MCMC_SP.proposedValues(jj),Model_SP.prior.values{jj}(1),Model_SP.prior.values{jj}(2));
            PP_prop = log(f); % convert to log prior (PP)
            % calculate assuming the current parameter values
            f = Model_SP.prior.equations{jj}(MCMC_SP.currentValues(jj),Model_SP.prior.values{jj}(1),Model_SP.prior.values{jj}(2));
            PP_cur = log(f);

            % calculate the corrected acceptance ratio -------------------------------------
            WLR = exp((LL_prop+PP_prop)-(LL_cur+PP_cur)+LC); % weighted likelihood ratio
            A = min([1,WLR]); % acceptance criterion
            r = rand(1,1); % random draw from uniform distribution
            if r <= A % if random draw less than acceptance
                MCMC_SP.decisionHx(nn,jj) = 1; % accept proposal
                MCMC_SP.currentValues(jj) = MCMC_SP.proposedValues(jj); % make curent position the proposed
            else
                % decisionHx(1,jj) = 0; % reject proposal
                % this is not used since initialization of matrix is zeros
            end
            MCMC_SP.History(MCMC_SP.counter,jj) = MCMC_SP.currentValues(jj);  % update the sampling history
        end
        MCMC_SP.counter = MCMC_SP.counter +1; % update the iteration counter


     % Now run for Combined ------------------
        for jj=1:Model_C.likelihood.nParameters 
            % start with all proposed parameter values equal to the current parameter values
            MCMC_C.proposedValues = MCMC_C.currentValues; 
            
            % get new proposal parameter value ---------------------------------
            pd1 = MCMC_C.proposal.distributions{jj}(MCMC_C.currentValues(jj),MCMC_C.proposalTuning(jj)); % create distribution object using current values
            MCMC_C.proposedValues(1,jj) = pd1.random(1); % randomly draw one sample from the distribution and update the proposed values with it
            % account for assymmetry in the proposal distributions (metropolis-hastings)
            pd2 = MCMC_C.proposal.distributions{jj}(MCMC_C.proposedValues(jj),MCMC_C.proposalTuning(jj)); % create distribution object using proposed values
            C = pd1.pdf(MCMC_C.currentValues(jj)) / pd2.pdf(MCMC_C.proposedValues(jj)); % ratio gives a correction value (C) account for assymmetry 
            LC = log(C); % convert to log space (LC) so compatible with log likelihoods and log priors

            % calculate the log-likelihoods -----------------------------------
            % calculate assuming the proposed parameter values
            II = length(d);
            f1 = zeros(II,1);
            f2 = zeros(II,1);
            for ii=1:II
                x = [d(ii);dSP(ii)];
                % make the copula and express as bivariate standard normal
                if Model_SP.likelihood.nParameters == 2
                    muGamma = MCMC.currentValues(1);
                    sigmaGamma = MCMC.currentValues(2);
                    pd_SP = makedist('normal',muGamma,sigmaGamma);
                    %
                    muGamma = MCMC_SP.currentValues(1);
                    sigmaGamma = MCMC_SP.currentValues(2);
                    pd_AP = makedist('normal',muGamma,sigmaGamma);
                elseif Model_SP.likelihood.nParameters == 3
                    muEpsilon = MCMC.currentValues(1);
                    sigmaEpsilon = MCMC.currentValues(2);
                    nuEpsilon = MCMC.currentValues(3);
                    pd_AP = makedist('tlocationscale',muEpsilon,sigmaEpsilon,nuEpsilon);
                    %
                    muEpsilon = MCMC_SP.currentValues(1);
                    sigmaEpsilon = MCMC_SP.currentValues(2);
                    nuEpsilon = MCMC_SP.currentValues(3);
                    pd_SP = makedist('tlocationscale',muEpsilon,sigmaEpsilon,nuEpsilon);
                end
                C = [pd_AP.cdf(x(1)) pd_SP.cdf(x(2))];
                Z = norminv(C);
                    
                f1(ii,1) = Model_C.likelihood.equation{1}(Z(:),MCMC_C.proposedValues(1)); % likelihood of the data
                % calculate mu assuming the current values
                f2(ii,1) = Model_C.likelihood.equation{1}(Z(:),MCMC_C.currentValues(1));
            end
            LL_prop = sum(log(f1)); % convert to log likelihood (LL)
            LL_cur = sum(log(f2));

            % get the log of the prior ----------------------------------------
            % calculate assuming the proposed parameter values
            f = Model_C.prior.equations{jj}(MCMC_C.proposedValues(jj),Model_C.prior.values{jj}(1),Model_C.prior.values{jj}(2));
            PP_prop = log(f); % convert to log prior (PP)
            % calculate assuming the current parameter values
            f = Model_C.prior.equations{jj}(MCMC_C.currentValues(jj),Model_C.prior.values{jj}(1),Model_C.prior.values{jj}(2));
            PP_cur = log(f);

            % calculate the corrected acceptance ratio -------------------------------------
            WLR = exp((LL_prop+PP_prop)-(LL_cur+PP_cur)+LC); % weighted likelihood ratio
            A = min([1,WLR]); % acceptance criterion
            r = rand(1,1); % random draw from uniform distribution
            if r <= A % if random draw less than acceptance
                MCMC_C.decisionHx(nn,jj) = 1; % accept proposal
                MCMC_C.currentValues(jj) = MCMC_C.proposedValues(jj); % make curent position the proposed
            else
                % decisionHx(1,jj) = 0; % reject proposal
                % this is not used since initialization of matrix is zeros
            end
            MCMC_C.History(MCMC_C.counter,jj) = MCMC_C.currentValues(jj);  % update the sampling history
        end
        MCMC_C.counter = MCMC_C.counter +1; % update the iteration counter

    end
    MCMC.counter = MCMC.counter - 1; % correct for the last increment
    MCMC_SP.counter = MCMC_SP.counter - 1;
    MCMC_C.counter = MCMC_C.counter - 1;
    MCMC.AR = sum(MCMC.decisionHx,1)./MCMC.counter; % the overall acceptance rates for each parameter
    MCMC_SP.AR = sum(MCMC_SP.decisionHx,1)./MCMC_SP.counter;
    MCMC_C.AR = sum(MCMC_C.decisionHx,1)./MCMC_C.counter;

    MCMC.History(1:MCMC.burnInSamples,:) = []; % strip off the burn-in samples
    MCMC_SP.History(1:MCMC_SP.burnInSamples,:) = [];
    MCMC_C.History(1:MCMC_C.burnInSamples,:) = [];
    if MCMC.thinning > 1 % do this if thinning of the chain was requested
        [nRows,nCols] = size(MCMC.History);
        for ii=1:nCols
            x = MCMC.History(:,ii);
            X = reshape(x,MCMC.thinning,nRows/MCMC.thinning);
            x = X(1,:)';
            XX(:,ii) = x; % keep only the requested amount
        end
        MCMC.History = XX; % write the thinned history to the MCMC structure
        MCMC.nSamples = origSamples; % final number of samples after burn-in and thinning is the original number requested

        [nRows,nCols] = size(MCMC_SP.History);
        for ii=1:nCols
            x = MCMC_SP.History(:,ii);
            X = reshape(x,MCMC_SP.thinning,nRows/MCMC_SP.thinning);
            x = X(1,:)';
            XX(:,ii) = x; % keep only the requested amount
        end
        MCMC_SP.History = XX; % write the thinned history to the MCMC structure
        MCMC_SP.nSamples = origSamples; % final number of samples after burn-in and thinning is the original number requested

        [nRows,nCols] = size(MCMC_C.History);
        for ii=1:nCols
            x = MCMC_C.History(:,ii);
            X = reshape(x,MCMC_C.thinning,nRows/MCMC_C.thinning);
            x = X(1,:)';
            XX(:,ii) = x; % keep only the requested amount
        end
        MCMC_C.History = XX; % write the thinned history to the MCMC structure
        MCMC_C.nSamples = origSamples; % final number of samples after burn-in and thinning is the original number requested
    end
    MCMC = unstandardize(MCMC,Model); % put coefficients back into natural units
    MCMC_SP = unstandardize(MCMC_SP,Model_SP);
    MCMC_C.Chain(:,1) = MCMC_C.History(:,1);
    MCMC_AP = MCMC;
    Model_AP = Model;
end

% INTERNAL FUNCTIONS ------------------------------------------------------
function [Model] = standardize(Model)
    % Standardize the data (both independent and dependent variables)
    % Here, we use z-scores.
    % Original data are denoted x and y. Standardized data in both cases
    % are denoted by z. 
    x =  Model.data.dependent.y;
    m = mean(x);
    s = std(x);
    for ii=1:size(m,2)
        z(:,ii) = (x(:,ii)-m(ii))/s(ii);
    end
    Model.data.dependent.z = z;
    % save the mean and standard deviation values so can unstandardize at the end
    Model.data.dependent.mz = m; % mean for z-score
    Model.data.dependent.sz = s; % standard deviation for z-score
 end
function [MCMC] = unstandardize(MCMC,Model)
    % Convert model parameters back to original data units of measure
    % The unstandardized history is saved as MCMC.Chain.
    m = Model.data.dependent.mz; % mean for z-score, dependent variable
    s = Model.data.dependent.sz; % standard deviation for z-score
   
    z1 = MCMC.History(:,1); % mu 
    z2 = MCMC.History(:,2); % sigma 
    if size(MCMC.History,2)==3
        z3 = MCMC.History(:,3); % nu 
    end

    MCMC.Chain(:,1) = (z1*s + m);
    MCMC.Chain(:,2) = z2*s;
   
    if size(MCMC.History,2)==3
        MCMC.Chain(:,3) = z3; % nu does not get scaled!
    end
end
function [MCMC] = adjustTuning(MCMC)
    % adjust the proposal distribution tuning to target 50% acceptance rate overall.
    n = length(MCMC.proposalTuning); % number of parameters
    nn = MCMC.counter; % number of iterations completed so far
    for ii=1:n % adjust separately for each parameter
        ar = sum(MCMC.decisionHx(1:nn,ii))/nn; % current acceptance rate
        % adjust tuning value based on current acceptance rate (ar)
        if ar > 0.6 % ar is too large; make propTuning larger
            MCMC.proposalTuning(ii) = MCMC.proposalTuning(ii) + (MCMC.proposalTuning(ii)*0.2);
        elseif ar < 0.4 % ar is too small, make propTuning smaller
            MCMC.proposalTuning(ii) = MCMC.proposalTuning(ii) - (MCMC.proposalTuning(ii)*0.2);
        end
    end
end
