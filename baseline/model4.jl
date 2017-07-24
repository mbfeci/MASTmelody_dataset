function acc(model, data)
    sumloss = numloss = 0

    for d in data

        refT = d["RefSegsTrue"];
        perT = d["PerSegsTrue"];
        perF = d["PerSegsFalse"];

        for ref in refT
            for per in perT
                dc = reshape(KnetArray(vcat(per',ref')), (2,1000,1,1))
                z = predict(model, dc)
                sumloss += mean((z .* 1) .> 0)
                numloss += 1
            end
        end

        for ref in refT
            for per in perF
                dc = reshape(KnetArray(vcat(per',ref')), (2,1000,1,1))
                z = predict(model, dc)
                sumloss += mean((z .* -1) .> 0)
                numloss += 1
            end
        end
    end
    sumloss / numloss
end

function init_weights(wsize)
    model = Array(Any, 4)

    model[1] = KnetArray(0.1*randn(Float32, 2,wsize,1,1))
    model[2] =  KnetArray(Float32,1,1,1,1);

    model[end-1] = KnetArray(0.1*randn(Float32,1000-wsize+1,1))
    model[end] = KnetArray(zeros(Float32,1,1))
    #model[end] = xavier(vocab,embed)    #We (word embedding vector)
    return model
end

function loss(w,d,z)
    ypred = predict(w,d)
    lf = log(1 .+ exp(-z .* ypred))
    sum(lf)/length(lf)
end

function predict(w, dc)
    cr = relu(conv4(w[1], dc).+w[2]);
    reshape(cr,(1,951))*w[end-1] .+ w[end];
end

eucgrad = grad(loss);

function train!(m, data, opts)
    count = 0
    for d in data
        refT = d["RefSegsTrue"];
        perT = d["PerSegsTrue"];
        perF = d["PerSegsFalse"];

        for ref in refT
            for per in perT
                dc = reshape(KnetArray(vcat(per',ref')), (2,1000,1,1))
                dw = eucgrad(m, dc, 1)
                for i in 1:length(m)
                    update!(m[i], dw[i], opts[i])
                end
            end
        end

        for ref in refT
            for per in perF
                dc = reshape(KnetArray(vcat(per',ref')), (2,1000,1,1))
                dw = eucgrad(m, dc, -1)
                for i in 1:length(m)
                    update!(m[i], dw[i], opts[i])
                end
            end
        end
        count += 1
        #println(count)
    end
end

function init_params(model)
    prms = Array(Any, length(model))
    for i in 1:length(model)
        prms[i] = Adam(;lr=0.00001)
    end
    return prms
end

function modelrun(data; epochs=100)
    model = init_weights(50)
    opts = init_params(model);

    println("Initialized model")
    println("Accuracies for: train-dev sets:")

    msg(e) = println((e,map(d->acc(model,d),data)...)); 
    msg(0)

    for epoch = 1:epochs
      # Alternative: one could keep the model with highest accuracy in development set results
      # and return that one instead of the last model
        train!(model, data[1], opts)#training on the train set (data[1])
        msg(epoch)
    end
    return model
end