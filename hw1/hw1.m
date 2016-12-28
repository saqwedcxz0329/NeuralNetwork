function hw1()
    X = [];
    Y_head = [];
    
    [x, y_head] = generateRandomDate(0.2, 1.8, 0.2, 1.8, 1);
    X = [X x];
    Y_head = [Y_head y_head];
    [x, y_head] = generateRandomDate(2.2, 3.8, 0.2, 1.8,0);
    X = [X x];
    Y_head = [Y_head y_head];
    [x, y_head] = generateRandomDate(4.2, 5.8, 0.2, 1.8,1);
    X = [X x];
    Y_head = [Y_head y_head];
    [x, y_head] = generateRandomDate(0.2,1.8,2.2,3.8,1);
    X = [X x];
    Y_head = [Y_head y_head];
    [x, y_head] = generateRandomDate(2.2,3.8,2.2,3.8,1);
    X = [X x];
    Y_head = [Y_head y_head];
    [x, y_head] = generateRandomDate(4.2,5.8,2.2,3.8,0);
    X = [X x];
    Y_head = [Y_head y_head];
    [x, y_head] = generateRandomDate(0.2,1.8,4.2,5.8,0);
    X = [X x];
    Y_head = [Y_head y_head];
    [x, y_head] = generateRandomDate(2.2,3.8,4.2,5.8,1);
    X = [X x];
    Y_head = [Y_head y_head];
    [x, y_head] = generateRandomDate(4.2,5.8,4.2,5.8,1);
    X = [X x];
    Y_head = [Y_head y_head];
    
%     [X, Y_head] = parseData('hw1data.dat');
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
end

function [X, Y_head] = generateRandomDate(x0, x1, y0, y1, value)
    x = rand(1, 100) * (x1-x0) + x0;
    y = rand(1, 100) * (y1-y0) + y0;
    X = [x; y];
    Y_head = zeros(1, 100);
    Y_head = Y_head + value;
end


% function initWeights(layers)
%     global weightList;
%     weightList = cell(1, length(layers)-1);
%     for i = 2:length(layers)
%         Wij = [];
%         for j = 1: layers(i)
%             Weights = (rand(1, layers(i-1) + 1)' - 0.5);
%             Wij = [Wij Weights];
%         end
%         weightList{i-1} = Wij;
%     end
%     disp(weightList);
% end

function initWeights(layers)
    global weightList;
    weightList = cell(1, length(layers)-1);
    Wij = [1 1 0 0; 0 0 1 1; -2 -4 -2 -4];
    weightList{1} = Wij;
    for i = 3:length(layers)
        Wij = [];
        for j = 1: layers(i)
            Weights = (rand(1, layers(i-1) + 1)' - 0.5);
            Wij = [Wij Weights];
        end
        weightList{i-1} = Wij;
    end
    disp(weightList);
end

function training(X, Y_head)
    
    global weightList;
    cur = pwd;
    disp(cur);
    file_name = fopen('C:\Users\Louis\Documents\NeuralNetwork\hw1\error.txt', 'w');
    for j = 1 : 250
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
        end
        disp(sigma);
        fprintf(file_name, '%f\n', sigma);
    end
    fclose(file_name);
    
    X_test = [1 3 5 1 3 5 1 3 5; 1 1 1 3 3 3 5 5 5];
     [~, c] = size(X_test);
    for i = 1 : c
        Xi = X_test(:, i);
        [y, netList] = neuralOutput(Xi);
        for i =  1: length(netList)
            disp(netList{i})
            disp('======')
        end
    end
    
    
%     plot(X(1, find(Y_head == 0)), X(2, find(Y_head == 0)), 'x', X(1, find(Y_head == 1)), X(2, find(Y_head == 1)), 'o');
%     for weights = weightList{1}
%         a = 0 : 1: 6;
%         %a = 0: 0.1: 1;
%         w1 = weights(1);
%         w2 = weights(2);
%         w3 = weights(3);
%         b = ((-w1*a) - w3)/w2;
%         hold on;
%         plot(a, b, 'r');
%         ylim([0 6]);
%         %ylim([0 1]);
%         hold off;
%     end

end

function updateWeights(netList, deltaList)
    global weightList;
    learning_rate = 0.5;
    for i = length(weightList) : -1 : 2
        Net_i = netList{i};
        Delta = deltaList{i};
        
        W = weightList{i};
        Delta = Delta';
        A = learning_rate .* [Net_i;1] * Delta;
        W = W + A;
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
        Netj = 1 ./ (1 + exp(-Z)); 
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
    for i = length(weightList) - 1 : -1 : 2
        Net_i = netList{i};
        Net_j = netList{i+1};
        W = weightList{i+1};
        [w_r, ~] = size(W);
        Delta = W(1:w_r-1, :) * Delta .* Net_j .* (1.0-Net_j);

%         git  = [];
%         for j = 1 : length(Net_j)
%             net_j = Net_j(j);
%             W_jk = getRowW(i+1,j);
%             delta = (W_jk * Delta) * net_j * (1.0-net_j);
%             delta_j = [delta_j delta];
%         end
%         Delta = delta_j';
        deltaList{i} = Delta;
    end
end

function error = getError(y, y_head)
	error = (y - y_head).^2;
end

function W = getColW(layer, index)
	global weightList;
	W = weightList{layer};
	W = W(:, index);
end

function W = getRowW(layer, index)
    global weightList;
    W = weightList{layer};
	W = W(index, :);
end