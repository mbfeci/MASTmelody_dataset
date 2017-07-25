function acc(w, data)
    sumloss = numloss = 0

    for d in data
        x = d[1];
        y = d[2];

        z = model(w, x)

        sumloss += mean((z .* y) .> 0)
        numloss += 1
    end
    sumloss/numloss
end

function init_weights(size)
    model = Array(Any, 2)

    model[end-1] = 0.1*randn(2*size,1)
    model[end] = zeros(1,1)        

    #model[end] = xavier(vocab,embed)    #We (word embedding vector)
    return model
end

function loss(w,x,z)
    ypred = model(w,x)
    mean(log(1 .+ exp(-z .* ypred)))
end

function model(w,x)
   x*w[end-1] .+ w[end]
end

lgrad = grad(loss);

function train!(m, data, opts)
    count = 0
    for d in data
        x = d[1];
        y = d[2];


        dw = lgrad(m, x, y)

        for i in 1:length(m)
            update!(m[i], dw[i], opts[i])
        end

        count += 1
        #println(count)
    end
end

function init_params(model)
    prms = Array(Any, length(model))
    for i in 1:length(model)
        prms[i] = Adam()
    end
    return prms
end

function modelrun(data; epochs=100)
    w = init_weights(1000)
    opts = init_params(w);

    println("Initialized model")
    println("Accuracies for: train-dev sets:")

    msg(e) = println((e,map(d->acc(w,d),data)...)); 
    msg(0)

    for epoch = 1:epochs
      # Alternative: one could keep the model with highest accuracy in development set results
      # and return that one instead of the last model
        train!(w, data[1], opts)#training on the train set (data[1])
        msg(epoch)
    end
    return model
end