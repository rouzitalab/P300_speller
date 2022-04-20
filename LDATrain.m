function W = LDATrain(Class1, Class2)

N1 = size(Class1,1);
N2 = size(Class2,1);
N = N1 + N2;

u1 = mean(Class1,1);
u2 = mean(Class2,1);

Sigma1 = cov(Class1);
Sigma2 = cov(Class2);

Sigma = (N1/N)*Sigma1 + (N2/N)*Sigma2;

W = Sigma\(u1-u2)';
% size(u1)
% size(u2)
% size(Sigma1)
% size(Sigma2)
% size(Sigma)

end