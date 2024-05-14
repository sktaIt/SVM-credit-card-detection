data = readtable('creditcard.csv');

X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));

cv = cvpartition(size(X, 1), 'HoldOut', 0.2);

X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

svmModel = fitcsvm(X_train, y_train);

y_pred = predict(svmModel, X_test);

index = {'non-fraud', 'fraud'};
figure;
gscatter(X_train(:, 1), X_train(:, 2), y_train);
hold on;
legend(index, 'Location', 'best');

sv = svmModel.SupportVectors;

plot(sv(:, 1), sv(:, 2), 'ko', 'MarkerSize', 10);

x1range = min(X_train(:, 1)) - 1 : 0.1 : max(X_train(:, 1)) + 1;
x2range = min(X_train(:, 2)) - 1 : 0.1 : max(X_train(:, 2)) + 1;
[x1, x2] = meshgrid(x1range, x2range);
XGrid = [x1(:), x2(:)];

K=100;
numTestInstances = size(X_test, 1);
predictedLabels = zeros(numTestInstances, 1);

for i = 1:numTestInstances
    distances = sqrt(sum((X_train - X_test(i, :)).^2, 2));
    
    [~, indices] = sort(distances);
    knnIndices = indices(1:K);
    
    knnLabels = y_train(knnIndices);
    
    labelCounts = tabulate(knnLabels);
    
    [~, maxCountIndex] = max(labelCounts(:, 2));
    predictedLabels(i) = labelCounts(maxCountIndex, 1);
end

tp = sum(predictedLabels(y_test == 1) == 1);  % True positive
fp = sum(predictedLabels(y_test == 0) == 1);  % False positive
tn = sum(predictedLabels(y_test == 0) == 0);  % True negative
fn = sum(predictedLabels(y_test == 1) == 0);  % False negative

precision = tp / (tp + fp);  % Precision
recall = tp / (tp + fn);  % Recall
f1_score = 2 * (precision * recall) / (precision + recall);  % F1-score
accuracy = sum(predictedLabels == y_test) / numel(y_test);  % Accuracy

mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
[~, ~, ~, AUC] = perfcurve(y_test, predictedLabels, 1);

fprintf('Recall: %.2f\n', recall);
fprintf('F1-score: %.2f\n', f1_score);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('AUC: %.3f\n', AUC);
fprintf('Matthews Correlation Coefficient (MCC): %.2f\n', mcc);
