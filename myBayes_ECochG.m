function [Model,MCMC] = myBayes_ECochG()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Model,MCMC] = myBayes_ECochG;
%
% Calculate Bayesian statistics for the paper entitled,
% "Minimum Detectable Differences in Electrocochleography Measurements: 
% Bayesian-based Predictions", by Shawn S. Goodman, Jeffery T Lichtenhan, 
% and Skyler Jennings.
%
% This code creates the Model and MCMC structures.
% It calls an MCMC engine (myBayes_ECochG_engine.m) to solve for the posteriors.
% It also calculates the posterier distributions and HDIs, and plots results.
%
% REQUIRED FUNTIONS and FILES:
% myBayes_ECochG_engine.m
% evaluateChainConvergence.m
% fullBiDataSet.mat
% Machine Learning and Statistics Toolbox
%
% OUTPUT ARGUMENTS:
% Model = model structure (data type: struct), including the data and the 
%           likelihood, prior, and posterior distributions.
% MCMC = (data type: struct) MCMC sampling results.
%
% Author: Shawn Goodman
% Date: January 23-August 24, 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    % USER MODIFIABLE PARAMETERS ------------------------------------------
    whichOne = 'epsilon'; % Choose from 'gamma', or 'epsilon'
                          % Run once for gamma, and again for epsilon
    
    nSamples = 30000; % number of samples in each chain
    burnInSamples = 1000; % number of burn-in samples to discard
    thinning = 3; % thin the chains to reduce the autocorrelation between 
                    % adjacent samples. Reduce by this amount (take every ith sample, i=thinning)

    runParallel = 1; % run on parallel processor or not. Faster, but requires toolbox
    nChains = 4; % Number of chain to run, each with new pseudorandom starting values 
                 % Chains will be combined for one final posterior estimate

    dataPath = 'C:\myWork\ARLas\Peripheral\analysis\SJ\SkylerData\'; % location of data set
    dataFileName = 'fullBiDataSet.mat'; % name of data set file
    savePath = 'C:\myWork\ARLas\Peripheral\analysis\SJ\'; % location where the data should be saved
    %--------------------------------------------------------------------------


    % get the data
    dummy = load([dataPath,dataFileName]);
    if strcmp(whichOne,'gamma')
        xAP = dummy.GAMMA.AP; % data for AP (compound action potential)
        xSP = dummy.GAMMA.SP; % data for SP (summating potential)
    elseif strcmp(whichOne,'epsilon')
        xAP = dummy.EPSILON.AP;
        xSP = dummy.EPSILON.SP;
        starts = dummy.starts;
        finishes = dummy.finishes;
        m = dummy.n; % number of repeated measurements made for each subject;
                     % can be used to weight the likelihoods.
    end 
    
    % Make three models: one for SP, one for AP, one for combining via copula
    % ---------------------------------------------------------------------
    tic
    disp(['RUNNING:',whichOne])
    disp(' ')

% MODEL for AP ------------------------------------------------------------
    MCMC.nSamples = nSamples; % final number of samples desired in each chain
    MCMC.burnInSamples = burnInSamples; % discared this number of initial samples
    MCMC.thinning = thinning; % keep every ith value (1 keeps all samples, 2 keeps every other, 3 keeps every third, etc.)
    Model.data.dependent.y = xAP; % dependent variable
    % -----------------------------
    if strcmp(whichOne,'gamma')
        %Model.likelihood.weighting = getWeights; % weights are square root of number of repeated measurements for each participant, normalized to sum to 1
        % NOTE: Set the following to empty set [] to run a standard
        % analysis. These can be altered to run weighted likelihood.
        Model.likelihood.weighting = []; % if empty, will use standard likelihood
        Model.likelihood.starts = [];
        Model.likelihood.finishes = [];
        Model.likelihood.m = [];
        % Set up the likelihoods:
        f1_Likelihood = @(x,mu,sigma) (1./(sigma*sqrt(2*pi))) .* exp(-((x-mu).^2)./(2*sigma.^2)); % normal likelihood
        Model.likelihood.equation = {f1_Likelihood}; % save the likelihood equation
        Model.likelihood.name = {'normal'}; % name the likelihood distribution
        Model.likelihood.nParameters = 2; % total number of parameters in the likelihood
        Model.likelihood.parameterNames = {'mu','sigma'}; % name of the parameters to be estimated
        Model.likelihood.startingValues = [0,.8]; % pick reasonable (normalized) starting values for the parameters
    elseif strcmp(whichOne,'epsilon')
        Model.likelihood.weighting = [];
        Model.likelihood.starts = starts; % vector of indices showing where each participant starts and finishes
        Model.likelihood.finishes = finishes;
        Model.likelihood.m = m; % number of repeats for each subject
        % Set up the t distribution likelihood:
            f_Likelihood = @(x,mu,sigma,nu) ((gamma((nu+1)/2))/(gamma(nu/2)*sqrt(pi*nu)*sigma) .* (1+(1/nu)*((x-mu)/sigma).^2).^(-(nu+1)/2)); % the likelihood function equation
        Model.likelihood.equation = {f_Likelihood}; % save the likelihood equation
        Model.likelihood.name = {'t'}; % name the likelihood distribution
        Model.likelihood.nParameters = 3; % total number of parameters in the likelihood
        Model.likelihood.parameterNames = {'mu','sigma','nu'}; % name of the parameters to be estimated
        Model.likelihood.startingValues = [0,.9,7]; % pick reasonable (normalized) starting values for the parameters
    end
    % -----------------------------------------------
    if strcmp(whichOne,'gamma')   
        % Assign a prior distribution for each parameter of normal distribution:
        % For the normal likehood, we have 2 parameters: mu, and sigma.
        % mu can take any real number, so broad normal distributions are good priors.
        % sigma must be any real positive number, and small values are more likely, so half-Cauchy is a good prior
        f_prior1 = @(theta,m,s) (1./(s*sqrt(2*pi))).*exp(-0.5*((m-theta)./s).^2); % equation for prior for first parameter
        f_prior2 = @(theta,m,s) (2./(pi*s)).*(1./(1+(m-abs(theta)).^2./s.^2)); % equation for prior for second parameter
        Model.prior.equations = {f_prior1,f_prior2}; % save the prior equations
        Model.prior.names = {'normal','halfCauchy'}; % name of the prior distributions
        Model.prior.values = {[0,50],[0,2]}; % assign the parameter values of the prior distributions
    elseif strcmp(whichOne,'epsilon') 
        % Assign a prior distribution for each parameter of t distribution:
        % Here we have 3 parameters: mu, sigma, and nu. 
        % mu can take any real number, so broad normal distributions are good priors
        % sigma must be any real positive number, and small values are more likely, so half-Cauchy is a good prior
        % Use a gamma distribution for nu.
        f_prior1 = @(theta,m,s) (1./(s*sqrt(2*pi))).*exp(-0.5*((m-theta)./s).^2); % equation for prior for first parameter
        f_prior2 = @(theta,m,s) (2./(pi*s)).*(1./(1+(m-abs(theta)).^2./s.^2)); % equation for prior for second parameter
        f_prior3 = @(theta,a,b) ((theta.^(a-1).*exp(-b*theta).*b^a)./gamma(a)); % equation for prior for third parameter
        Model.prior.equations = {f_prior1,f_prior2,f_prior3}; % save the prior equations
        Model.prior.names = {'normal','halfCauchy','gamma'}; % name of the prior distributions
        Model.prior.values = {[0,2],[0,1],[3,0.25]}; % assign the parameter values of the prior distributions
    end
    % ------------------------------------------------
    if strcmp(whichOne,'gamma')
        % Assign proposal distributions for each parameter:
        % Here we have 2 parameters: mu and sigma. 
        % mu can take any real number, so normal distributions are good proposals.
        % sigma must be any positive real number, so truncated normal is a good proposal.
        % Use anonymous functions create the distribution objects.
        % In the MCMC sampler, proposal values will be created by drawing from the objects.
        pd_prop1 = @(mu,sigma) makedist('Normal','mu',mu,'sigma',sigma); % proposal distribution object for b0
        pd_prop2 = @(mu,sigma) truncate(makedist('Normal','mu',mu,'sigma',sigma),0,100); % proposal distribution object for sigma
        MCMC.proposal.distributions = {pd_prop1,pd_prop2};
        MCMC.proposal.names = {'normal','truncatedNormal'};
        MCMC.proposalTuning = [.3, .2]; % set starting tuning parameters; these will be adjusted as needed
    elseif strcmp(whichOne,'epsilon')
        % Assign proposal distributions for each parameter:
        % Here we have 3 parameters: mu, sigma, and nu. 
        % mu can take any real number, so normal distributions are good proposals.
        % sigma and nu must be any positive real number, so truncated normal is a good proposal.
        % Use anonymous functions create the distribution objects.
        % In the sampler, proposal values will be created by drawing from the objects.
        pd_prop1 = @(mu,sigma) makedist('Normal','mu',mu,'sigma',sigma); % proposal distribution object for mu
        pd_prop2 = @(mu,sigma) truncate(makedist('Normal','mu',mu,'sigma',sigma),0,100); % proposal distribution object for sigma 
        pd_prop3 = @(mu,sigma) truncate(makedist('Normal','mu',mu,'sigma',sigma),0,100); % proposal distribution object for nu
        MCMC.proposal.distributions = {pd_prop1,pd_prop2,pd_prop3};
        MCMC.proposal.names = {'normal','truncatedNormal','truncatedNormal'};
        MCMC.proposalTuning = [.3, .2, 10];
    end
    MCMC_AP = MCMC;
    Model_AP = Model;
    clear MCMC Model

% MODEL for SP ------------------------------------------------------------
    MCMC.nSamples = nSamples; % final number of samples in each chain
    MCMC.burnInSamples = burnInSamples; % discared this number of initial samples
    MCMC.thinning = thinning; % keep every ith value (1 keeps all samples, 2 keeps every other, 3 keeps every third, etc.)
    Model.data.dependent.y = xSP; % dependent variable
    % -----------------------------
    if strcmp(whichOne,'gamma')
        % Set up the likelihoods:
            f1_Likelihood = @(x,mu,sigma) (1./(sigma*sqrt(2*pi))) .* exp(-((x-mu).^2)./(2*sigma.^2)); % normal likelihood
        Model.likelihood.equation = {f1_Likelihood}; % save the likelihood equation
        Model.likelihood.name = {'normal'}; % name the likelihood distribution
        Model.likelihood.nParameters = 2; % total number of parameters in the likelihood
        Model.likelihood.parameterNames = {'mu','sigma'}; % name of the parameters to be estimated
        Model.likelihood.startingValues = [0,.8]; % pick reasonable (normalized) starting values for the parameters
    elseif strcmp(whichOne,'epsilon') 
        % Set up the t distribution likelihood:
            f_Likelihood = @(x,mu,sigma,nu) ((gamma((nu+1)/2))/(gamma(nu/2)*sqrt(pi*nu)*sigma) .* (1+(1/nu)*((x-mu)/sigma).^2).^(-(nu+1)/2)); % the likelihood function equation
        Model.likelihood.equation = {f_Likelihood}; % save the likelihood equation
        Model.likelihood.name = {'t'}; % name the likelihood distribution
        Model.likelihood.nParameters = 3; % total number of parameters in the likelihood
        Model.likelihood.parameterNames = {'mu','sigma','nu'}; % name of the parameters to be estimated
        Model.likelihood.startingValues = [0,.9,10]; % pick reasonable (normalized) starting values for the parameters
    end
    % -----------------------------------------------
    if strcmp(whichOne,'gamma')
        % Assign a prior distribution for each parameter of normal distribution:
        % For the normal likehood, we have 2 parameters: mu, and sigma.
        % mu can take any real number, so broad normal distributions are good priors.
        % sigma must be any real positive number, and small values are more likely, so half-Cauchy is a good prior
        f_prior1 = @(theta,m,s) (1./(s*sqrt(2*pi))).*exp(-0.5*((m-theta)./s).^2); % equation for prior for first parameter
        f_prior2 = @(theta,m,s) (2./(pi*s)).*(1./(1+(m-abs(theta)).^2./s.^2)); % equation for prior for third parameter
        Model.prior.equations = {f_prior1,f_prior2}; % save the prior equations
        Model.prior.names = {'normal','halfCauchy'}; % name of the prior distributions
        Model.prior.values = {[0,50],[0,2]}; % assign the parameter values of the prior distributions
    elseif strcmp(whichOne,'epsilon')    
        % Assign a prior distribution for each parameter of t distribution:
        % Here we have 3 parameters: mu, sigma, and nu. 
        % mu can take any real number, so broad normal distributions are good priors
        % sigma must be any real positive number, and small values are more likely, so half-Cauchy is a good prior
        f_prior1 = @(theta,m,s) (1./(s*sqrt(2*pi))).*exp(-0.5*((m-theta)./s).^2); % equation for prior for first parameter
        f_prior2 = @(theta,m,s) (2./(pi*s)).*(1./(1+(m-abs(theta)).^2./s.^2)); % equation for prior for second parameter
        f_prior3 = @(theta,a,b) ((theta.^(a-1).*exp(-b*theta).*b^a)./gamma(a)); % equation for prior for third parameter
        Model.prior.equations = {f_prior1,f_prior2,f_prior3}; % save the prior equations
        Model.prior.names = {'normal','halfCauchy','gamma'}; % name of the prior distributions
        Model.prior.values = {[0,2],[0,1],[3,0.25]}; % assign the parameter values of the prior distributions
    end
    % ------------------------------------------------
    if strcmp(whichOne,'gamma')
        % Assign proposal distributions for each parameter:
        % Here we have 2 parameters: mu and sigma. 
        % mu can take any real number, so normal distributions are good proposals.
        % sigma must be any positive real number, so truncated normal is a good proposal.
        % Use anonymous functions create the distribution objects.
        % In the sampler, proposal values will be created by drawing from the objects.
        pd_prop1 = @(mu,sigma) makedist('Normal','mu',mu,'sigma',sigma); % proposal distribution object for mu
        pd_prop2 = @(mu,sigma) truncate(makedist('Normal','mu',mu,'sigma',sigma),0,100); % proposal distribution object for sigma
        MCMC.proposal.distributions = {pd_prop1,pd_prop2};
        MCMC.proposal.names = {'normal','truncatedNormal'};
        MCMC.proposalTuning = [.3, .2];
    elseif strcmp(whichOne,'epsilon')
        % Assign proposal distributions for each parameter:
        % Here we have 3 parameters: mu, sigma, and nu. 
        % mu can take any real number, so normal distributions are good proposals.
        % sigma and nu must be any positive real number, so truncated normal is a good proposal.
        % Here, the anonymous functions create the distribution objects.
        % In the sampler, proposal values will be created by drawing from the objects.
        pd_prop1 = @(mu,sigma) makedist('Normal','mu',mu,'sigma',sigma); % proposal distribution object for mu
        pd_prop2 = @(mu,sigma) truncate(makedist('Normal','mu',mu,'sigma',sigma),0,100); % proposal distribution object for sigma 
        pd_prop3 = @(mu,sigma) truncate(makedist('Normal','mu',mu,'sigma',sigma),0,100); % proposal distribution object for nu
        MCMC.proposal.distributions = {pd_prop1,pd_prop2,pd_prop3};
        MCMC.proposal.names = {'normal','truncatedNormal','truncatedNormal'};
        MCMC.proposalTuning = [.3, .2, 10];
    end
    MCMC_SP = MCMC;
    Model_SP = Model;
    clear MCMC Model

% MODEL for Combined AP and SP------------------------------------------------------------
    MCMC.nSamples = nSamples; % final number of samples in each chain
    MCMC.burnInSamples = burnInSamples; % discared this number of initial samples
    MCMC.thinning = thinning; % keep every ith value (1 keeps all samples, 2 keeps every other, 3 keeps every third, etc.)
    Model.data.dependent.y = []; % dependent variable
    % -----------------------------
        % Set up the likelihoods:
            f1_Likelihood = @(x,rho) exp(-0.5*x'*inv([1 rho; rho 1])*x) ./ (sqrt((2*pi)^2.*det([1 rho; rho 1]))); % bivariate normal likelihood
        Model.likelihood.equation = {f1_Likelihood}; % save the likelihood equation
        Model.likelihood.name = {'bivariateNormal'}; % name the likelihood distribution
        Model.likelihood.nParameters = 1; % total number of parameters in the likelihood
        Model.likelihood.parameterNames = {'rho'}; % name of the parameters to be estimated
        Model.likelihood.startingValues = [.5]; % pick reasonable (normalized) starting values for the parameters
    % -----------------------------------------------
        % Assign a prior distribution for each parameter of normal distribution:
        % For the bivariate normal likelihood, we have 1 parameter: rho.
        % rho can be any real number between -1 and 1, so uniform on [-1 1] is a good prior.
           f_prior1 = @(theta,m,s) 0.5; % equation for prior for third parameter
        Model.prior.equations = {f_prior1}; % save the prior equations
        Model.prior.names = {'uniform'}; % name of the prior distributions
        Model.prior.values = {[0,0]}; % assign the parameter values of the prior distributions
    % ------------------------------------------------
        % Assign proposal distributions for each parameter:
        pd_prop1 = @(mu,sigma) truncate(makedist('Normal','mu',mu,'sigma',sigma),-1,1); % proposal distribution object for sigma
        MCMC.proposal.distributions = {pd_prop1};
        MCMC.proposal.names = {'truncatedNormal'};
        MCMC.proposalTuning = [0.26];
        
    MCMC_C = MCMC;
    Model_C = Model;
    clear MCMC Model

    %------------------------------------------------------------------------------------
    %------------------------------------------------------------------------------------

    % If you have the parallel computing toolbox and workers/multi-core processor, 
    % is ~2.5 times fastser to run 4 chains in parallel than in series.
    if runParallel==1
        disp(['Using Parallel Processing:'])
        disp(['Starting MCMC Chain ',num2str(1),' of ',num2str(nChains)])
        F1 = parfeval(@myBayes_ECochG_engine,6,Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C);
        disp(['Starting MCMC Chain ',num2str(2),' of ',num2str(nChains)])
        F2 = parfeval(@myBayes_ECochG_engine,6,Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C);
        disp(['Starting MCMC Chain ',num2str(3),' of ',num2str(nChains)])
        F3 = parfeval(@myBayes_ECochG_engine,6,Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C);
        disp(['Starting MCMC Chain ',num2str(4),' of ',num2str(nChains)])
        F4 = parfeval(@myBayes_ECochG_engine,6,Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C);
        
        disp(['Finished MCMC Chain ',num2str(1),' of ',num2str(nChains)])
        [Model_AP1,MCMC_AP1,Model_SP1,MCMC_SP1,Model_C1,MCMC_C1] = fetchOutputs(F1);
        disp(['Finished MCMC Chain ',num2str(2),' of ',num2str(nChains)])
        [Model_AP2,MCMC_AP2,Model_SP2,MCMC_SP2,Model_C2,MCMC_C2] = fetchOutputs(F2);
        disp(['Finished MCMC Chain ',num2str(3),' of ',num2str(nChains)])
        [Model_AP3,MCMC_AP3,Model_SP3,MCMC_SP3,Model_C3,MCMC_C3] = fetchOutputs(F3);
        disp(['Finished MCMC Chain ',num2str(4),' of ',num2str(nChains)])
        [Model_AP4,MCMC_AP4,Model_SP4,MCMC_SP4,Model_C4,MCMC_C4] = fetchOutputs(F4);
        cancel(F1); cancel(F2); cancel(F3); cancel(F4)
    else
        disp(['Using Serial Processing:'])
        disp(['Running MCMC Chain ',num2str(1),' of ',num2str(nChains)])
        [Model_AP1,MCMC_AP1,Model_SP1,MCMC_SP1,Model_C1,MCMC_C1] = myBayes_ECochG_engine(Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C); % call the MCMC engine
        disp(['Running MCMC Chain ',num2str(2),' of ',num2str(nChains)])
        [Model_AP2,MCMC_AP2,Model_SP2,MCMC_SP2,Model_C2,MCMC_C2] = myBayes_ECochG_engine(Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C); % call the MCMC engine
        disp(['Running MCMC Chain ',num2str(3),' of ',num2str(nChains)])
        [Model_AP3,MCMC_AP3,Model_SP3,MCMC_SP3,Model_C3,MCMC_C3] = myBayes_ECochG_engine(Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C); % call the MCMC engine
        disp(['Running MCMC Chain ',num2str(4),' of ',num2str(nChains)])
        [Model_AP4,MCMC_AP4,Model_SP4,MCMC_SP4,Model_C4,MCMC_C4] = myBayes_ECochG_engine(Model_AP,MCMC_AP,Model_SP,MCMC_SP,Model_C,MCMC_C); % call the MCMC engine
    end

    % Combine results into one big chain
    MCMC_AP = MCMC_AP1;
    ii=1;
    bigChain1 = [MCMC_AP1.Chain(:,ii),MCMC_AP2.Chain(:,ii),MCMC_AP3.Chain(:,ii),MCMC_AP4.Chain(:,ii)];
    ii=2;
    bigChain2 = [MCMC_AP1.Chain(:,ii),MCMC_AP2.Chain(:,ii),MCMC_AP3.Chain(:,ii),MCMC_AP4.Chain(:,ii)];
    if strcmp(whichOne,'epsilon')
        ii=3;
        bigChain3 = [MCMC_AP1.Chain(:,ii),MCMC_AP2.Chain(:,ii),MCMC_AP3.Chain(:,ii),MCMC_AP4.Chain(:,ii)];
        MCMC_AP.Chain = [bigChain1(:),bigChain2(:),bigChain3(:)];
    else
        MCMC_AP.Chain = [bigChain1(:),bigChain2(:)];
    end
    
    MCMC_SP = MCMC_SP1;
    ii=1;
    bigChain1 = [MCMC_SP1.Chain(:,ii),MCMC_SP2.Chain(:,ii),MCMC_SP3.Chain(:,ii),MCMC_SP4.Chain(:,ii)];
    ii=2;
    bigChain2 = [MCMC_SP1.Chain(:,ii),MCMC_SP2.Chain(:,ii),MCMC_SP3.Chain(:,ii),MCMC_SP4.Chain(:,ii)];
    if strcmp(whichOne,'epsilon')
        ii=3;
        bigChain3 = [MCMC_SP1.Chain(:,ii),MCMC_SP2.Chain(:,ii),MCMC_SP3.Chain(:,ii),MCMC_SP4.Chain(:,ii)];
        MCMC_SP.Chain = [bigChain1(:),bigChain2(:),bigChain3(:)];
    else
        MCMC_SP.Chain = [bigChain1(:),bigChain2(:)];
    end
    
    MCMC_C = MCMC_C1;
    ii=1;
    bigChain1 = [MCMC_C1.Chain(:,ii),MCMC_C2.Chain(:,ii),MCMC_C3.Chain(:,ii),MCMC_C4.Chain(:,ii)];
    MCMC_C.Chain = bigChain1(:);

    % calculate posterior distributions from combined chains
    posterior1_AP = calculatePosterior(MCMC_AP.Chain(:,1)); % for mu
    posterior2_AP = calculatePosterior(MCMC_AP.Chain(:,2)); % for sigma
    if strcmp(whichOne,'epsilon')
        posterior3_AP = calculatePosterior(MCMC_AP.Chain(:,3)); % for nu degrees of freedom
    end
    posterior1_SP = calculatePosterior(MCMC_SP.Chain(:,1)); % for mu
    posterior2_SP = calculatePosterior(MCMC_SP.Chain(:,2)); % for sigma
    if strcmp(whichOne,'epsilon')
        posterior3_SP = calculatePosterior(MCMC_SP.Chain(:,3)); % for nu degrees of freedom
    end
    posterior1_C = calculatePosterior(MCMC_C.Chain(:,1)); % for rho

    % save posterior distribution structures into the model structure
    Model_AP.Posterior1 = posterior1_AP;
    Model_AP.Posterior2 = posterior2_AP;
    if strcmp(whichOne,'epsilon')
        Model_AP.Posterior3 = posterior3_AP;
    end
    Model_SP.Posterior1 = posterior1_SP;
    Model_SP.Posterior2 = posterior2_SP;
    if strcmp(whichOne,'epsilon')
        Model_SP.Posterior3 = posterior3_SP;
    end
    Model_C.Posterior1 = posterior1_C;

    % Save the analysis ---------------------------------------------------
    if strcmp(whichOne,'gamma')
        saveName = 'SJ_bayesAnalysisGamma.mat';
    else
        saveName = 'SJ_bayesAnalysisEpsilon.mat';
    end
    %saveName = ARLas_saveName(savePath,saveName);
    save([savePath,saveName],'Model_AP1','MCMC_AP1','Model_SP1','MCMC_SP1','Model_C1','MCMC_C1',...)
        'Model_AP2','MCMC_AP2','Model_SP2','MCMC_SP2','Model_C2','MCMC_C2',...
        'Model_AP3','MCMC_AP3','Model_SP3','MCMC_SP3','Model_C3','MCMC_C3',...
        'Model_AP4','MCMC_AP4','Model_SP4','MCMC_SP4','Model_C4','MCMC_C4',...
        'MCMC_AP','Model_AP','MCMC_SP','Model_SP','MCMC_C','Model_C')

    % plot results --------------------------------------------------------
    figure(1)
        plot(posterior1_AP.xx,posterior1_AP.yy,'k')
        ymax = max(posterior1_AP.yy);
        hold on
        line([posterior1_AP.hdi95(1),posterior1_AP.hdi95(1)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        line([posterior1_AP.hdi95(2),posterior1_AP.hdi95(2)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        xlabel('\mu_A_P (mean)')
    
    figure(2)
        plot(posterior1_SP.xx,posterior1_SP.yy,'k')
        ymax = max(posterior1_SP.yy);
        hold on
        line([posterior1_SP.hdi95(1),posterior1_SP.hdi95(1)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        line([posterior1_SP.hdi95(2),posterior1_SP.hdi95(2)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        xlabel('\mu_S_P (mean)')

    figure(3)
        plot(posterior1_C.xx,posterior1_C.yy,'k')
        ymax = max(posterior1_C.yy);
        hold on
        line([posterior1_C.hdi95(1),posterior1_C.hdi95(1)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        line([posterior1_C.hdi95(2),posterior1_C.hdi95(2)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        xlabel('\rho_\gamma (mean)')

    figure(4)
        plot(posterior2_AP.xx,posterior2_AP.yy,'k')
        ymax = max(posterior2_AP.yy);
        hold on
        line([posterior2_AP.hdi95(1),posterior2_AP.hdi95(1)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        line([posterior2_AP.hdi95(2),posterior2_AP.hdi95(2)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        xlabel('\sigma_A_P (stdev)')

    figure(5)
        plot(posterior2_SP.xx,posterior2_SP.yy,'k')
        ymax = max(posterior2_SP.yy);
        hold on
        line([posterior2_SP.hdi95(1),posterior2_SP.hdi95(1)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        line([posterior2_SP.hdi95(2),posterior2_SP.hdi95(2)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        xlabel('\sigma_S_P (stdev)')

    if strcmp(whichOne,'epsilon')

        figure(6)
        plot(posterior3_AP.xx,posterior3_AP.yy,'k')
        ymax = max(posterior3_AP.yy);
        hold on
        line([posterior3_AP.hdi95(1),posterior3_AP.hdi95(1)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        line([posterior3_AP.hdi95(2),posterior3_AP.hdi95(2)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        xlabel('\nu_A_P (deg freedom)')

        figure(7)
        plot(posterior3_SP.xx,posterior3_SP.yy,'k')
        ymax = max(posterior3_SP.yy);
        hold on
        line([posterior3_SP.hdi95(1),posterior3_SP.hdi95(1)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        line([posterior3_SP.hdi95(2),posterior3_SP.hdi95(2)],[0 ymax],'Color',[0 0 0],'LineWidth',0.5)
        xlabel('\nu_S_P (deg freedom)')
    end


    for ii=1:size(MCMC_AP1.Chain,2)
        Chains = [MCMC_AP1.Chain(:,ii),MCMC_AP2.Chain(:,ii),MCMC_AP3.Chain(:,ii),MCMC_AP4.Chain(:,ii)];
        evaluateChainConvergence(Chains);
        title(['AP Parameter ',num2str(ii)])
    end
    
    for ii=1:size(MCMC_SP1.Chain,2)
        Chains = [MCMC_SP1.Chain(:,ii),MCMC_SP2.Chain(:,ii),MCMC_SP3.Chain(:,ii),MCMC_SP4.Chain(:,ii)];
        evaluateChainConvergence(Chains);
        title(['SP Parameter ',num2str(ii)])
    end
    
    for ii=1:size(MCMC_C.Chain,2)
        Chains = [MCMC_C1.Chain(:,ii),MCMC_C2.Chain(:,ii),MCMC_C3.Chain(:,ii),MCMC_C4.Chain(:,ii)];
        evaluateChainConvergence(Chains);
        title(['C Parameter ',num2str(ii)])
    end
    
    toc/60
    keyboard
end

% INTERNAL FUNCTIONS ------------------------------------------------------
function [w] = getWeights()
    w = [  1.7321
    2.0000
    2.0000
    2.0000
    2.8284
    3.1623
    2.2361
    1.7321
    2.6458
    2.4495
    2.2361
    2.0000
    3.4641
    2.2361
    2.4495
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142
    1.4142];
    w = w./sum(w);
end
function [post] = calculatePosterior(thetaHx)
    dist = fitdist(thetaHx,'kernel');
    minx = min(thetaHx);
    maxx = max(thetaHx);
    N = 1000;
    xx = linspace(minx,maxx,N);
    yy = dist.pdf(xx);
    
    alpha = 0.05;
    [hdi95,HDIdensity,inOut95] = getHDI(alpha,xx,yy);
    alpha = 0.5;
    [hdi50,HDIdensity,inOut50] = getHDI(alpha,xx,yy);
    med = dist.median;
    [~,medIndx] = min(abs(med-xx));
    medY = yy(medIndx);
    [modeY,qq] = max(yy);
    mode = xx(qq);

    post.dist = dist;
    post.hdi95 = hdi95;
    post.hdi50 = hdi50;
    post.med = med;
    post.medY = medY;
    post.mode = mode;
    post.modeY = modeY;
    post.xx = xx;
    post.yy = yy;
    post.inOut95 = inOut95;
    post.inOut50 = inOut50;
end
function [HDI,HDIdensity,inOut] = getHDI(alpha,xx,yy)
    % Calculate highest density interval on alpha
    [p,I] = sort(yy,'ascend'); % sort densities in ascending order
    cutValue = sum(p)*alpha; % find the alpha% cut value
    cutIndx = min(find(cumsum(p)>cutValue)); % find the location of the cut value
    waterline = p(cutIndx); % this is the cutoff density
    [goodValues,goodIndx] = find(yy >= waterline); % locate all values > cut
    HDI = [xx(min(goodIndx));xx(max(goodIndx))]; % determine the interval
    HDIdensity = waterline;
    inOut = (yy >= waterline);
end
