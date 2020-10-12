import numpy as np
#import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import scipy.spatial.distance as scp
import json

def convert_to_codewords(vocab, name):
    asd = np.load(name)
    kp = asd["keypoints"]
    d = asd["descriptors"]
    #print(vocab.shape, d.shape, scp.cdist(d, vocab).shape)
    dists = scp.cdist(d, vocab)

    #print(np.argmin(dists, axis=1).shape)
    tmp = vocab[np.argmin(dists, axis=1),:]
    return kp, tmp

def image_detect_and_compute(img_name, vocab=[], method="r2d2"):
    img = cv.imread(img_name)
    if not img_name[-5:] == ".r2d2":
        img_name += ".r2d2"
    if not len(vocab):
        data = np.load(img_name)
        return img, data["keypoints"], data["descriptors"]
    kp, d = convert_to_codewords(vocab, img_name)
    return img, kp, d

def search(img1, kp1, des1, path2, vocab=[], draw=False):
    
    #img1, kp1, des1 = image_detect_and_compute(path1, vocab=vocab)
    img2, kp2, des2 = image_detect_and_compute(path2, vocab=vocab)
    nmatches = 20
    #print(des1.dtype)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance) # Sort matches by distance.  Best come first.
    #print(matches[0].distance)
    if not draw:
        return sum([x.distance for x in matches[:20]]) #matches[0].distance 
    kp1 = [cv.KeyPoint(k[0], k[1], k[2]) for k in kp1]
    kp2 = [cv.KeyPoint(k[0], k[1], k[2]) for k in kp2]
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], img2, flags=2) # Show top 10 matches
    plt.figure(figsize=(16, 16))
    #plt.title(type(detector))
    plt.imshow(img_matches); plt.show()

def run(path):
    tmp = {}
    tmp["q"] = path
    vocab = []
    best_d = [np.inf for _ in range(5)]
    best_i = ["" for _ in range(5)]
    m = 0
    #path = "q/landmarks-Query-100.jpg"
    img1, kp1, des1 = image_detect_and_compute(path, vocab=vocab)
    db = os.listdir("db/")
    db = np.random.choice(db, 100)
    path_ = path.split("/")[-1].split("-")
    correct =  path_[0] + "-Reference-" + path_[-1]
    if not correct in db:
        db[0] = correct
    for i in db:

        i = "db/" + i
        if not i[-5:] == ".r2d2":
            continue
        #if not "landmark" in i:
        #    continue
        m += 1
        #if not "landmarks" in i:
        #    continue
        d = search(img1, kp1, des1, i, vocab=vocab)
        n = 0
        for j, d_ in enumerate(best_d):
        
            if d < d_:
                best_d.insert(j, d)
                best_i.insert(j, i)
                best_d = best_d[:5]
                best_i = best_i[:5]
                break
            
        
                
    #print(best_d, best_i)
    tmp["r"] = best_i
    return tmp

if __name__=="__main__":
    results = []
    result_path = "results.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = json.load(f)
    past_work = [x["q"] for x in results]    
    for i in os.listdir("q/"):
        i = "q/" + i
        if i in past_work:
            continue
        if not i[-5:] == ".r2d2":
            continue
        if not "landmark" in i:
            continue
        results.append(run(i))
        with open(result_path, "w") as f:
            json.dump(results, f)

    
