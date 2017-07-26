
for p in ("JSON","JLD","DSP","Knet")
    if Pkg.installed(p) == nothing; Pkg.add(p); end
end

using Knet
include("model6.jl")

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

function minibatch2(data; batch = 100, splt=((200,200), (73,73)))
  tout = Float32[];#tout: true-out: matching melodic pairs' (piano-singing) feature
  fout = Float32[];#fout: false-out: non-matching melodic pairs' (piano-singing) feature
  ref = Any[];
  idx =  Array{Any,1}();
  count = 1;
  for d in data
    refT = d["RefSegsTrue"];#in julia string indexes can be safely used :)
    perT = d["PerSegsTrue"];
    perF = d["PerSegsFalse"];

    #Pair all reference segments with performance segments marked as true/pass
    # compute the feature vector and the add to true-pairs-data
    
    #refT = refT[randperm(length(refT))[1:5]]
    for j=1:length(perT)#all true-performance recordings
        # Computation of the feature as histogram of differences after matching two signals by DTW
        # the vector is appended to true-output pair
        # If you would prefer another input for the MLP, consider modifying this line
        append!(tout, perT[j]');
        push!(idx, (count, count+length(refT)))
    end

    #pair all reference segments with performance segments marked as false/fail
    # compute the feature vector and the add to false-pairs-data
    for j=1:length(perF)#all false-performance recordings
        #computation of the feature as histogram of differences after matching two signals by DTW
        # the vector is appended to false-output pair
        # If you would prefer another input for the MLP, consider modifying this line
        append!(fout, perF[j]');
        push!(idx, (count, count+length(refT)))
    end

    for i=1:length(refT)
        push!(ref, refT[i]')
    end
    println(length(refT))
    count += length(refT);
  end

  #re-shape feature vectors computed from true and false pairs in matrix form and return both
  rows=1000;
  cols = div(length(tout),rows);
  tout = reshape(tout, (cols,rows));
  cols = div(length(fout),rows);
  fout = reshape(fout, (cols,rows));

  (nt,nd) = size(tout);#nt: number of true-pairs
  (nf,nd) = size(fout);#nf: number of false-pairs
  rt = randperm(nt);#forming random order of indexes to shuffle
  rf = randperm(nf);
  nt = nf = 0
  bdata = Any[];#batch data for all sets of splits
  for (t,f) in splt
    if(t>0 && f>0)#if there exists (0,0) in splits, skip it
      #forming input-vector for MLP
      # Left half contains true samples, right half contains false samples
      x = vcat(tout[rt[nt+1:nt+t],:], fout[rf[nf+1:nf+f],:]);
      #forming the output vector: 1:true, -1: false
      y = vcat(ones(Float32,1,t), -ones(Float32,1,f))

      nt += t; nf += f;
      r = randperm(t+f);
      batches = Any[];
      #forming a single batch in each loop, putting in 'batches'
      for i=1:batch:length(r)-batch
        xbatch = x[r[i:i+batch-1],:];
        ybatch = y[r[i:i+batch-1],:];
        push!(batches, (xbatch, ybatch))
      end
      #adding to 'batches' for this split in 'bdata'
      push!(bdata, batches);
    end
  end
  return bdata
  
end


function minibatch3(data; batch = 50, splt=((3000,3000),))
  tout = Float32[];#tout: true-out: matching melodic pairs' (piano-singing) feature
  fout = Float32[];#fout: false-out: non-matching melodic pairs' (piano-singing) feature

  for d in data
    refT = d["RefSegsTrue"];#in julia string indexes can be safely used :)
    perT = d["PerSegsTrue"];
    perF = d["PerSegsFalse"];

    #Pair all reference segments with performance segments marked as true/pass
    # compute the feature vector and the add to true-pairs-data
    
    refT = refT[randperm(length(refT))[1:22]]

    for i=1:length(refT)
        for j=1:length(perT)#all true-performance recordings
            # Computation of the feature as histogram of differences after matching two signals by DTW
            # the vector is appended to true-output pair
            # If you would prefer another input for the MLP, consider modifying this line
            append!(tout, vcat(perT[j],refT[i]));
        end
    end

    for i=1:length(refT)
        for j=1:length(perF)#all false-performance recordings
            #computation of the feature as histogram of differences after matching two signals by DTW
            # the vector is appended to false-output pair
            # If you would prefer another input for the MLP, consider modifying this line
            append!(fout, vcat(perF[j],refT[i]));
        end
    end
  end

  #re-shape feature vectors computed from true and false pairs in matrix form and return both
  rows=2000;
  cols = div(length(tout),rows);
  tout = reshape(tout, (rows,cols));
  cols = div(length(fout),rows);
  fout = reshape(fout, (rows,cols));

  tout = tout'
  fout = fout'

  (nt, nd) = size(tout);#nt: number of true-pairs
  (nf, nd) = size(fout);#nf: number of false-pairs
  rt = randperm(nt);#forming random order of indexes to shuffle
  rf = randperm(nf);
  nt = nf = 0
  bdata = Any[];#batch data for all sets of splits
  for (t,f) in splt
    if(t>0 && f>0)#if there exists (0,0) in splits, skip it
      #forming input-vector for MLP
      # Left half contains true samples, right half contains false samples
      x = vcat(tout[rt[nt+1:nt+t],:], fout[rf[nf+1:nf+f],:]);
      #forming the output vector: 1:true, -1: false
      y = vcat(ones(Float32,t,1), -ones(Float32,f,1))

      nt += t; nf += f;
      r = randperm(t+f);
      batches = Any[];
      #forming a single batch in each loop, putting in 'batches'
      for i=1:batch:length(r)-batch+1
        xbatch = x[r[i:i+batch-1],:];
        ybatch = y[r[i:i+batch-1],:];
        push!(batches, (xbatch, ybatch))
      end
      #adding to 'batches' for this split in 'bdata'
      push!(bdata, batches);
    end
  end
  return bdata
end

function decideSplits(pairData,option)
  if option==1
    (tempX,tempY)=pairData;
    minNumPairs=min(size(tempX,2),size(tempY,2));
    splitRatio=0.9;#split for train and development
    splts=((Int(floor(minNumPairs*splitRatio)),Int(floor(minNumPairs*splitRatio))),(Int(floor(minNumPairs*(1-splitRatio))),Int(floor(minNumPairs*(1-splitRatio)))));
    return splts;
  elseif option==2
    (tempX,tempY)=pairData;
    #minimum of number of true-pairs and false-pairs will be used for training
    minNumPairs=min(size(tempX,2),size(tempY,2));
    splts=((minNumPairs,minNumPairs),(0,0));
    return splts;
  end
end

function runTests(data)

    #=
    spdata = splitData(data)
    trn = minibatch3(spdata[1]);
    tst = minibatch3(spdata[2]; splt = ((500,500),));
    =#

    println("Data splitted.")

    #spdata = map(x->minibatch(x,batchsize), spdata)
    #println("Minibatch completed.")
    modelrun(data)


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
