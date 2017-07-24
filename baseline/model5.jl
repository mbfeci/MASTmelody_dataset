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
    model = Array(Any, 10)

    model[1] = KnetArray(0.01*randn(Float32, 2,3,1,2))
    model[2] =  KnetArray(Float32,1,1,1,1);

    model[3] = KnetArray(0.01*randn(Float32, 1,3,2,4))
    model[4] =  KnetArray(Float32,1,1,1,1);

    model[5] = KnetArray(0.01*randn(Float32, 1,3,4,8))
    model[6] =  KnetArray(Float32,1,1,1,1);

    model[7] = KnetArray(0.01*randn(Float32, 1,3,8,16))
    model[8] =  KnetArray(Float32,1,1,1,1);

    model[end-1] = KnetArray(0.01*randn(Float32,16*60,1))
    model[end] = KnetArray(zeros(Float32,1,1))
    #model[end] = xavier(vocab,embed)    #We (word embedding vector)
    return model
end

function cbfp(w,x;start=1,padding=0)
    pool(relu(conv4(w[start],x;padding=padding).+w[start+1]))
end

function loss(w,d,z)
    ypred = predict(w,d)
    lf = log(1 .+ exp(-z .* ypred))
    sum(lf)/length(lf)
end

function predict(w, dc)
    x1 = cbfp(w,dc;start=1);
    x2 = cbfp(w,x1;start=3);
    x3 = cbfp(w,x2;start=5);
    x4 = cbfp(w,x3;start=7);

    reshape(x4,(1,960))*w[end-1] .+ w[end];
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