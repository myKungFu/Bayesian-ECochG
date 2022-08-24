function [] = readme_ECochG()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate Bayesian statistics for the paper entitled,
% "Minimum Detectable Differences in Electrocochleography Measurements: 
% Bayesian-based Predictions", by Shawn S. Goodman, Jeffery T Lichtenhan, 
% and Skyler Jennings.
%
% This repository contains 5 Matlab (*.m) files and 3 data (*.mat) files.
% The Bayesian analysis was performed using myBayes_ECochG.m.
% Calculation of dmin was performed using myBayes_ECochG_predictive.m.
%
% Bayesian analysis uses a Metropolis-Hastings MCMC sampler, implemented 
% using component-wise sampling. This code was written to promote understanding,
% without an emphasis on speed.
% 
% myBayes_ECochG.m was called twice, once for gamma and once for epsilon.
% Using a 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz   3.30 GHz
% processor with 40 GB RAM, parallel processing times for 
% 4 chains of 10,000 samples with 1000 sample burn-in and 1/3 thinning:
% gamma = 30 minutes; epsilon = 96 minutes.
% 
% All code written by Shawn Goodman
% All code in this repository licensed under GNU General Public License v3.0.
% Code written and tested using MATLAB 2021b 
%  '9.11.0.1809720 (R2021b) Update 1'
% along with the Distributed Computing Toolbox 
% and the Machine Learning and Statistics Toolbox
% Address questions to shawn-goodman@uiowa.edu
% 
% Author: Shawn Goodman, PhD
% Auditory Research Lab, Director
% The University of Iowa
% Iowa City, IA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
