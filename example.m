% Demo script

data = load('MGtimeseries.mat');   
data = data.MGtimeseries;
inputData = cell2mat(data(1:end-1))'; 
targetData = cell2mat(data(2:end))';

washout = 100;

trlen = 2000; tslen = 2000; 
trX{1} = inputData(1:trlen);
tsX{1} = inputData(trlen+1:trlen+tslen);
% Remove initial points from target!
trY = targetData(1+washout:trlen);
tsY = targetData(trlen+1+washout:trlen+tslen);

esn = ESN(50, 'leakRate', 0.3, 'spectralRadius', 0.5, 'regularization', 1e-8);

esn.train(trX, trY, washout);

output = esn.predict(tsX, washout);

error = immse(output, tsY);
fprintf('Test error: %g\n', error);

plot(1:length(output), output, 1:length(tsY), tsY);
legend('Output', 'Target');