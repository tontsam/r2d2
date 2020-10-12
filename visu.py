import numpy as np
#import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import scipy.spatial.distance as scp


def convert_to_codewords(vocab, name):
    asd = np.load(name)
    kp = asd["keypoints"]
    d = asd["descriptors"]
    #print(vocab.shape, d.shape, scp.cdist(d, vocab).shape)
    dists = scp.cdist(d, vocab)

    #print(np.argmin(dists, axis=1).shape)
    tmp = vocab[np.argmin(dists, axis=1),:]
    return kp, tmp

def image_detect_and_compute(img_name, vocab=[]):
    img = cv.imread(img_name)
    if not len(vocab):
        data = np.load(img_name + ".r2d2")
        return img, data["keypoints"], data["descriptors"]
    kp, d = convert_to_codewords(vocab, img_name + ".r2d2")
    return img, kp, d

def run(path1, path2, vocab=[], draw=False):
    
    img1, kp1, des1 = image_detect_and_compute(path1, vocab=vocab)
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


if __name__=="__main__":
    #vocab = np.load("vocab.npy")
    vocab = []
    best_d = [np.inf for _ in range(5)]
    best_i = ["" for _ in range(5)]
    for i in os.listdir("mmsys-b/test/"):
        i = "mmsys-b/test/" + i
        if i[-5:] == ".r2d2":
            continue
        
        
        d = run("mmsys-b/train/landmarks-100.jpg", i, vocab=vocab)
        n = 0
        for j, d_ in enumerate(best_d):
        
            if d < d_:
                best_d.insert(j, d)
                best_i.insert(j, i)
                best_d = best_d[:5]
                best_i = best_i[:5]
                break
            
        
                
    print(best_d, best_i)
