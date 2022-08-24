function [] = evaluateChainConvergence(Chain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [] = evaluateChainConvergence(Chain);
%
% Calculate Bayesian statistics for the paper entitled,
% "Minimum Detectable Differences in Electrocochleography Measurements: 
% Bayesian-based Predictions", by Shawn S. Goodman, Jeffery T Lichtenhan, 
% and Skyler Jennings.
%
% INPUT ARGUMENTS:
% Chain = matrix of chains, each chain in a column.
% Use 4 or more chains by default.
%
% Analysis based on "Rank-normalization, folding, and localization: An 
% improved Rhat for assessing convergence of MCMC, 
% By Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, and Paul-Christian Bürkner
% arXiv:1903.08008v5 [stat.CO] 22 Jun 2021
%
% Following Vehtari et al., 
% Use Rhat based on rank-normalizing and folding the posterior draws, 
%       only using the sample if Rhat < 1.01. 
% Roughly speaking, the effective sample size of a quantity of interest captures 
% how many independent draws contain the same amount of information as the 
% dependent sample obtained by the MCMC algorithm. The higher the ESS the better. 
% Roughly speaking, the effective sample size of a quantity of interest captures 
%
% A small value of Rhat is not enough to ensure that an MCMC sample is useful in 
% % practice (Vats and Knudson, 2018). The effective sample size must also be 
% large enough to get stable inferences for quantities of interest.
% 
% Compute the ESS of a sample from a rank-normalized version of the quantity 
% of interest, using the rank transformation followed by the inverse normal 
% transformation. This is still indicative of the effective sample size for 
% computing an average. To ensure reliable estimates of variances and 
% autocorrelations needed for Rb and ESS, recommend requiring that the 
% rank-normalized ESS is greater than 400, a number we chose based on practical 
% experience and simulations as typically sufficient to get a stable estimate 
% of the Monte Carlo standard error.
%
% BOTTOM LINE: Look for Rhat < 1.01 and ESS > 400
%
% Example of code useage with results from myBayes_ECochG and
% myBayes_ECochG_engine.m:
% ii = 1;
% Chain = [MCMC_AP1.Chain(:,ii),MCMC_AP2.Chain(:,ii),MCMC_AP3.Chain(:,ii),MCMC_AP4.Chain(:,ii)];
%
% Author: Shawn Goodman
% Date: May 31-August 24, 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % To obtain a single conservative Rb estimate, report the maximum of 
    % rank normalized split-Rhat and rank normalized folded-split-Rhat for each parameter.
    %
    % Split Rhat:  Answers the question, "Did the chains mix well?"
    %
    % ESS: Answers the question, "Is the sample size large enough to get a stable
    % estimate of uncertainty?"
    %
    % Here we present split-Rhat, following Gelman et al. (2013) but using the notation of Stan Development Team (2018b).
    % This formulation represents the current standard in convergence diagnostics for iterative simulations. In the equations
    % below, 
    %
    % N = number of draws per chain, 
    % M = number of chains, 
    % S = M*N is the total number of draws from all chains, 
    %
    % θ (nm) is nth draw of mth chain, 
    % θ(.m) is the average of draws from mth chain, 
    % θ(..) is average of all draws. 
    % For each scalar summary of interest θ, we compute B and W, the between- and within-chain variances:
        
    if nargin == 0 % if there is no input, make four random number chains
        Chain = randn(10000,4); % four random chains
    end
    
    [I1_1,I1_2,I1_3,I1_4] = quantiles(Chain(:,1));
    [I2_1,I2_2,I2_3,I2_4] = quantiles(Chain(:,2));
    [I3_1,I3_2,I3_3,I3_4] = quantiles(Chain(:,3));
    [I4_1,I4_2,I4_3,I4_4] = quantiles(Chain(:,4));

    % Chain = matrix of chains, each chain in a column. Use 4 or more chains
    [NN,MM] = size(Chain); % original size of Chains
    if mod(NN,2)~=0 % make chain an even number of draws
        Chain(end,:) = [];
        NN = NN-1;
    end
    med = median(Chain(:));
    for m=1:MM
        %med = median(Chain(:,m));
        Zeta(:,m) = abs(Chain(:,m) - med);
    end

    Chains = [Chain(1:NN/2,:),Chain(NN/2+1:end,:)];
    Zeta = [Zeta(1:NN/2,:),Zeta(NN/2+1:end,:)];
    [N,M] = size(Chains);
    S = M*N; % total number of draws

    [splitRhat,Seff,SeffOld] = compute(Chains);

    % split-Rhat and Seff are well defined only if the marginal posteriors have finite 
    % mean and variance, therefore use rank normalized parameter values instead of the 
    % actual parameter values for the purpose of diagnosing convergence.
    chain = Chains(:);
    Ranks = tiedrank(chain);
    Z = norminv((Ranks-3/8)/(S+1/4));
    Z = reshape(Z,N,M);
    Ranks = reshape(Ranks,N*2,M/2);
    [splitRhatZ,SeffZ,SeffZOld] = compute(Z); % rank normalized split-Rhat 

    % Both original and rank normalized splitRhat can be fooled if the chains 
    % have the same location but different scales. This can happen if one or 
    % more chains is stuck near the middle of the distribution. To alleviate 
    % this problem, compute a rank normalized split-Rhat statistic not only for 
    % the original draws θ, but also for the corresponding folded draws
    % ζ(mn), absolute deviations from the median
    % This measures convergence in the tails rather than in the bulk of the distribution. 
%     for m=1:M
%         med = median(Chains(:,m));
%         Zeta(:,m) = abs(Chains(:,m) - med);
%     end
    [splitRhatZeta,~,~] = compute(Zeta); % rank normalized folded split-Rhat 

    disp('Rhat: (split, split-rank-normalized, folded-split-rank-normalized)')
    [splitRhat,splitRhatZ,splitRhatZeta]
    disp('ESS: splot, split-rank-normalized')
    [Seff,SeffZ]
    disp('Old ESS:')
    [SeffOld]
    
% Rank plots. Extending the idea of using ranks instead of the original parameter values, we propose using rank plots
% for each chain instead of trace plots. Rank plots, such as Figure 6, are histograms of the ranked posterior draws (ranked
% over all chains) plotted separately for each chain. If all of the chains are targeting the same posterior, we expect the
% ranks in each chain to be uniform, whereas if one chain has a different location or scale parameter, this will be reflected
% in the deviation from uniformity. If rank plots of all chains look similar, this indicates good mixing of the chains. As
% compared to trace plots, rank plots don’t tend to squeeze to a fuzzy mess when used with long chains.
    figure
    xmax = round(size(Chain,1)/12);
    subplot(2,2,1)
    histogram(Ranks(:,1),20)
    ylim([0,xmax])
    subplot(2,2,2)
    histogram(Ranks(:,2),20)
    ylim([0,xmax])
    subplot(2,2,3)
    histogram(Ranks(:,3),20)
    ylim([0,xmax])
    subplot(2,2,4)
    histogram(Ranks(:,4),20)
    ylim([0,xmax])

end

% INTERNAL FUNCTIONS ------------------------------------------------------
function [splitRhat,Seff,SeffOld] = compute(Chains)
    
    % calculate split-Rhat --------------
    [N,M] = size(Chains);
    %S = M*N; % total number of draws
    
    barThetaDotm = mean(Chains,1); % mean of each chain
    barThetaDotDot = mean(Chains(:)); % mean across all draws combined
    B = (N/(M-1))*sum((barThetaDotm - barThetaDotDot).^2); % between-chain variance
    
    s2 = zeros(M,1);
    for m=1:M
        s2(m,1) = (1/(N-1))*sum((Chains(:,m) - barThetaDotm(m)).^2);
    end
    W = (1/M)*sum(s2); % within chain variance
    
    varHat = (((N-1)/N)*W) + ((1/N)*B); % marginal posterior variance is weighted average of W and B
    splitRhat = sqrt(varHat/W);
    

    % compute effective sample size
    rhotHatm = zeros(N,M);
    for m=1:M
        % q = xcorr(Chains(:,m),'unbiased');
        % rhotHatm(:,m) = q(1:N);
        %rhotHatm(:,m) = q(N:end) / N * s2(m);
        %rhotHatm(:,m) = q(N:end) / (N * s2(m));
        q = xcorr(Chains(:,m),'normalized');
        q = q(1:N);
        % q = q / (N-1);
        % q = q - median(q);
        % rhotHatm(:,m) = q;
        nn = (1:1:N)';
        rhotHatm(:,m) = q(1:N)./nn;
    end
    rhotHat = 1-((W-(1/M)*sum(rhotHatm,2))./varHat); % multi-chain rhot, Eq. 10
    cut = length(rhotHat); %/ 2; %70;
    Seff = round((N*M)./(1+2*sum(rhotHat(1:cut)))); % Eq. 11

    cutoff = 50;
    Seff_avg = round(Seff / M);
    if Seff_avg < cutoff
        %warning('Effective Sample Sizes (Seff) < 50!')
    end
    % Important: only rely on Rhat estimate to make decisions about the quality 
    % of the chain if each of the split chains has an average ESS estimate of at least 50. 
    SeffOld = round(M*N*(varHat/B));

end

% INTERNAL FUNCTIONS ------------------------------------------------------
function [I1,I2,I3,I4] = quantiles(chain)
    xx = sort(chain(:));
    N = length(xx); % number of observations
    q = 100 *(0.5:N-0.5)./N;
    xx = [min(xx); xx(:); max(xx)];
    q = [0 q 100];

    F1 = interp1(q,xx,25); % the first fourth, approx 25th precentile
    F2 = interp1(q,xx,50); % the first half, approx 50th precentile
    F3 = interp1(q,xx,75); % the third fourth, approx 75th percentile
    
    I1 = (chain<F1);
    I2 = (chain>=F1 & chain<F2);
    I3 = (chain>=F2 & chain<F3);
    I4 = (chain>=F3);
end

% OLD CODE ----------------------------------------------------------------
    %Seff = M*N*(varHat/B); % proposed by Gelman (2003)