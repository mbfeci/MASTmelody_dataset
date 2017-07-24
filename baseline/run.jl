
for p in ("JSON","JLD","DSP","Knet")
    if Pkg.installed(p) == nothing; Pkg.add(p); end
end

using Knet
include("model4.jl")

function splitData(data; opt = 2, train = 28, dev = 4, test = 8)
    perm = randperm(length(data))
    if(opt==1)
        train_set = data[perm[1:train]]
        val_set = data[perm[train+1:train+dev]]
        test_set = data[perm[train+dev+1:train+dev+test]]
        return train_set,val_set,test_set
    elseif opt==2
        train_set = data[perm[1:train+dev]]
        test_set = data[perm[train+dev+1:train+dev+test]]
        return train_set,test_set
    end
end


function minibatch(data, batchsize=100)
    result = Any[]
    for d in data

        refT = d["RefSegsTrue"];
        perT = d["PerSegsTrue"];
        perF = d["PerSegsFalse"];

        ref = refT[rand(1:length(refT))]

        perSize = min(length(perT), length(perF))

        num_of_batches = div(perSize, batchsize)

        permT = randperm(length(perT))
        permF = randperm(length(perF))

        for i in 1:batchsize:num_of_batches*batchsize
            per = Array(Float32, batchsize, length(refT[1]))
            y = hcat(ones(Float32,1,batchsize/2), -ones(Float32,1,batchsize/2))
            for j in 1:batchsize/2
                per[i+j-1,:] = perT[permT[i+j-1]]
            end
            for j in batchsize/2+1:batchsize
                per[i+j-1,:] = perF[permF[i+j-1]]
            end
            push!(result, (ref,per,y))
        end

    end
    return result
end


function runTests(data)

    spdata = splitData(data);
    println("Data splitted.")

    #spdata = map(x->minibatch(x,batchsize), spdata)
    #println("Minibatch completed.")
    modelrun(spdata)

end


# Setting database and Julia tools directories
# 'dbaDir' is the folder that contains all f0s.txt files
# 'toolsDir' is the folder that contains all julia scripts in this example
toolsDir=dirname(@__FILE__); #current directory assigned as the tools directory
dbaDir=replace(toolsDir,"baseline","f0data");

include("gatherMelSegsInAPool.jl");

if !isfile(joinpath(dbaDir,"groupedMelSegData.jld"))
    runDBAPrepProcess(dbaDir);
end

data = load(joinpath(dbaDir,"groupedMelSegData.jld"), "data");

runTests(data);
