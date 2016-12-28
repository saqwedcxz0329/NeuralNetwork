function hw1()
    [X, Y_head] = parseData('hw1data.dat');
    layers = [2 4 3 1];
    global layerLength;
    layerLength = length(layers);
    initWeights(layers);
    training(X, Y_head);
end

function [X, Y_head] = parseData(file_name)
    data = importdata(file_name);
    X = data(:, 1:2)';
    Y_head = data(:, 3)';
    for i = find(Y_head < 0);
        Y_head(i) = 0;
    end

    plot(X(1, find(Y_head == 0)), X(2, find(Y_head == 0)), 'x', X(1, find(Y_head == 1)), X(2, find(Y_head == 1)), 'o');
end

function initWeights(layers)
    global weightList;
    weightList = cell(1, length(layers)-1);
    for i = 2:length(layers)
        Wij = [];
        for j = 1: layers(i)
            Weights = (rand(1, layers(i-1) + 1)');
            Wij = [Wij Weights];
        end
        weightList{i-1} = Wij;
    end
    disp(weightList);
end

function training(X, Y_head)
    global weightList;
    index = 1;
    for j = 1 : 1000
        disp(j);
        sigma = 0;
        [~, c] = size(X);
        for i = 1 : c
            Xi = X(:, i);
            y_head = Y_head(i);
            [y, netList] = neuralOutput(Xi);
            error = getError(y, y_head);
            deltaList = hideToOut(y_head, netList);
            deltaList = inToHide(netList, deltaList);
            updateWeights(netList, deltaList);
            sigma = sigma + error;
            index = index + 1;
        end
        disp(sigma);
        index = index + 1;
    end
    
    X_test = [0.15 0.5 0.85 0.15 0.5 0.85 0.15 0.5 0.85; 0.15 0.15 0.15 0.5 0.5 0.5 0.85 0.85 0.85];
     [~, c] = size(X_test);
    for i = 1 : c
        Xi = X_test(:, i);
        [y, netList] = neuralOutput(Xi);
        for i =  1: length(netList)
            disp(netList{i})
            disp('======')
        end
    end
    
    
    for weights = weightList{1}
        a = 0 : 0.1: 1;
        w1 = weights(1);
        w2 = weights(2);
        w3 = weights(3);
        b = ((-w1*a) - w3)/w2;
        hold on;
        plot(a, b, 'r');
        ylim([0 1]);
        hold off;
    end

end

function updateWeights(netList, deltaList)
    global weightList;
    learning_rate = 0.5;
    for i = fliplr(1:length(weightList))
        Net_i = netList{i};
        Delta = deltaList{i};
        
        W = weightList{i};
        W = W + learning_rate .* [Net_i;1] * Delta';
        weightList{i} = W;
    end
end

function [y, netList] = neuralOutput(Neti)
    global weightList;
    global layerLength;
	netList = cell(1, layerLength);
    netList{1} = Neti;
    for i = 1 : length(weightList)
        W = weightList{i};
        Z = W' * [Neti; 1];
        Netj = sigmoid(Z); 
        netList{i+1} = Netj;
        Neti = Netj;
    end
    y = Neti;
end

function deltaList = hideToOut(Y_head, netList)
    global layerLength;
    deltaList = cell(1, layerLength - 1);
    index = length(netList);
    Netk = netList{index};
    Netj = netList{index - 1};
    for i = 1 : length(Netk)
        netk = Netk(i);
        y_head = Y_head(i);
        delta = (y_head - netk) * netk * (1.0 - netk);
    end
    deltaList{layerLength-1} = delta;
end

function deltaList = inToHide(netList, deltaList)
    global weightList;
    Delta = deltaList{length(deltaList)};
    for i = fliplr(1: length(weightList) - 1)
        Net_i = netList{i};
        Net_j = netList{i+1};
        
        W = weightList{i+1};
        [w_r, ~] = size(W);
        Delta = W(1:w_r - 1, :) * Delta .* Net_j .* (1.0-Net_j);

        deltaList{i} = Delta;
    end
end

function error = getError(y, y_head)
	error = (y - y_head).^2;
end

function value = sigmoid(z)
	value =  1 ./ (1 + exp(-z));
end
