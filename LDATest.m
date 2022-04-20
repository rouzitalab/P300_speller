function y = LDATest(W,testData)

tmp = W' * testData';

y = mean(tmp);

clear tmp;

end