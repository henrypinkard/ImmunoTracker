clear 

load('ActiveLearningUncertaintySampling');
load('ActiveLearningFStatRandom');

meanFStatRandom = mean(fStatRandom,2)';
seFStatRandom = std(fStatRandom') / sqrt(size(fStatRandom,2));
meanFStatUncertainty = mean(fStatUncertainty,2)';
seFStatUncertainty = std(fStatUncertainty') / sqrt(size(fStatUncertainty,2));

figure(1)
cutoff = 650;
linechartwithuncertainty(uncertaintySamplingNumExamples(1:10:cutoff), meanFStatUncertainty(1:10:cutoff), seFStatUncertainty(1:10:cutoff) )
hold on
linechartwithuncertainty(randomSamplingNumExamples, meanFStatRandom, seFStatRandom )
hold off
xlabel('# Labeled training examples')
ylabel('F statistic')
legend('Uncertainty sampling','','Random sampling')

% load('ActiveLearningUncertaintySampling3ensemble.mat')
% meanFStatUncertainty3 = mean(fStatUncertainty,2)';
% seFStatUncertainty3 = std(fStatUncertainty') / sqrt(size(fStatUncertainty,2));
% hold on
% linechartwithuncertainty(uncertaintySamplingNumExamples', meanFStatUncertainty3, seFStatUncertainty3 )
% hold off