function y = ASE(learn, decay, reward)
    global w x_vec e
    sigma = 0.01;
    noise = sigma * randn;
    
    y = Activation(w' * x_vec + noise);
    w = w + learn * reward * e;
    e = decay*e + (1-decay) * y * x_vec;
end

function y = Activation(x)
    if x >= 0
        y = 2;
    else
        y = 1;
    end
end