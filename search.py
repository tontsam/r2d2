import numpy as np
import os, sys
import scipy.spatial.distance as scp
import faiss
import json
from vocab import Voctree

def convert_to_codewords(vocab, name):
    asd = np.load(name)
    d = asd["descriptors"]
    #print(vocab.shape, d.shape, scp.cdist(d, vocab).shape)
    #dists = scp.cdist(d, vocab)
    tmp = np.zeros(vocab.ntotal)
    for i in d:
        i = np.expand_dims(i, axis=0)
        q = vocab.search(i)
        #print(q)
        tmp[q] += 1
    #print(np.argmin(dists, axis=1).shape)
        
    #tmp[np.arange(dists.shape[0]), np.argmin(dists, axis=1)] = 1
    return tmp

def create_ndx (vocab, path):
    n = -1
    ndx = None
    fs = os.listdir(path)
    files = []
    for i in fs:
        i = path + i
        if not i[-5:] == ".r2d2":
            continue
        if "landmark" in i:
            continue
        files.append(i[:-5])
        n += 1
        tmp = convert_to_codewords(vocab, i)
        #print(tmp.shape)
        tmp = get_tf(tmp)
        if not n:
            #ndx = np.sum(tmp, axis=0)
            ndx = tmp[..., np.newaxis]
        else:
            #asd = np.sum(tmp, axis=0)
            asd = tmp[..., np.newaxis]
            ndx = np.concatenate((ndx, asd),axis=1)
        #print(ndx)

    ndx = ndx.T
    return ndx, files

def get_df(ndx):

    freqs = np.sum(ndx, axis=0)
    
    freqs =  ndx.shape[0] / freqs
    freqs = np.log(freqs)
    return freqs

def get_tf(v):
    v = np.divide(v, v.shape[0])
    return v

def evaluate(results):
    classes = {}
    c1 = 0
    c5 = 0
    a = 0
    for i in results:
        a += 1
        q = i["query"]
        q = q.split("/")[-1].split(".")[0].split("-")
        for n, p in enumerate(i["preds"]):
            p = p.split("/")[-1].split(".")[0].split("-")
            if p[0] == q[0] and p[-1] == q[-1]:

                if not n:
                    c1 += 1
                c5 += 1
                break
    print("Top1: ", c1/a)
    print("Top5: ", c5/a)


if __name__=="__main__":
    #vocab = np.load("vocab.npy")
    vocab = Voctree(fpath="vocab.npz")
    #ndx_train, label_train = create_ndx(vocab, sys.argv[1] + "/train/")
    #ndx_test, label_test = create_ndx(vocab, sys.argv[1] + "/test/")
    ndx_train, label_train = create_ndx(vocab, "db/")
    df = get_df(ndx_train)
    ndx_test, label_test = create_ndx(vocab, "q/")
    ndx_train = get_tf(ndx_train)
    ndx_test = get_tf(ndx_test)
    ndx_train = np.multiply(ndx_train, df)
    ndx_test = np.multiply(ndx_test, df)
    print(ndx_train.shape)
    print(ndx_test.shape)
    #np.save("landmarks_train.npy", ndx_train)
    #np.save("landmarks_test.npy", ndx_test)
    index = faiss.IndexFlatIP(vocab.ntotal)
    index.add(np.ascontiguousarray(ndx_train).astype(np.float32))
    print(index.ntotal)
    D, I = index.search(np.ascontiguousarray(ndx_test).astype(np.float32), 5)
    print(I.shape)
    #query preds
    results = []
    for i, j in zip(label_test, I):
        tmp = {}
        tmp["query"] = i
        tmp["preds"] = []
        for k in j:
            tmp["preds"].append(label_train[k])
        results.append(tmp)
    with open("results.json", "w") as f:
        json.dump(results, f)
    evaluate(results)
