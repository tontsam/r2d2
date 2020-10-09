import numpy as np
import os, sys
from sklearn.cluster import KMeans, MiniBatchKMeans
import uuid

class Voctree:

    def __init__(self, dscr=np.array([[]]), centroids=[], names=[], structure={}, fpath=None, level=0, max_level=6, max_children=10, name="root"):


        self.leaves = []
        self.centroids = []
        self.leaf = (not len(centroids) and dscr.shape[0] < max_children) or (level >= max_level)
        self.dscr = dscr
        self.level = level
        self.max_level = max_level
        self.max_children = max_children
        self.name = name
        if fpath:
            self.load(fpath)
        elif not self.leaf:
            #print("Building at level ", level)
            self.build(centroids, names, structure)

    def build(self, centroids=[], names=[], structure={}):
        if len(centroids) > 0:
            #print(structure, type(structure), structure.item()["cd5d0de6"])
            if not len(structure):
                self.leaf = True
            for i in structure:
                
                #print(i)
                ndx = np.where(names == i)
                #print(centroids.shape, centroids[ndx].shape)
                self.centroids.append(centroids[ndx])
                #if len(structure[i]):
                #    self.leaf = True
                #    continue
                tmp = Voctree(centroids=centroids, names=names, structure=structure[i], level=self.level+1, max_level=self.max_level, max_children=self.max_children, name=i)
                self.leaves.append(tmp)
            if len(self.centroids):
                self.centroids = np.vstack(self.centroids)
        else:
            kmeans = MiniBatchKMeans(n_clusters=self.max_children)
            kmeans.fit(self.dscr)
            self.centroids = kmeans.cluster_centers_
            for i in range(len(self.centroids)):
                j = self.dscr[kmeans.labels_ == i]
                tmp = Voctree(j, level=self.level+1, max_level=self.max_level, max_children=self.max_children, name=str(uuid.uuid4())[:8])
                self.leaves.append(tmp)
            
    def save(self):
        c, n, s = self.represent()
    
        np.savez("vocab.npz", centroids=c, names=n, structure=s)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        centroids = data["centroids"]
    
        names = data["names"]
    
        structure = data["structure"].item()
        self.build(centroids, names, structure)
        
    def represent(self):
        ndxs = {}
        total_n = 0
        c = [] #clusters
        n = [] #names
        s = {} #structure
        c.append(self.centroids)
        for i in self.leaves:
            n.append(i.name)
        for i, (j, k) in enumerate(zip(self.leaves, self.centroids)):
            """
            if j.leaf:
                c.append(k)
                n.append(j.name)
                total_n += 1
                ndxs[str(i)] = {}
            else:
                ndxs_, j_ = j.represent()
            """
            #n.append(j.name)
            s[j.name] = {}
            if j.leaf:
                continue
            c_, n_, s_ = j.represent()
            if len(c_):
                c.append(c_.squeeze())
            for n__ in n_:
                n.append(n__)
            s[j.name] = s_
        if len(c):
            #for i in c:
            #    print(self.level*"\t", i.shape)
            c = np.vstack(c)
        return c, n, s


    def __str__(self):
        string = self.level*"\t"+" "+self.name
        for i,j in zip(self.leaves, self.centroids):
            string +=  str(j[-1]) + " " + str(i)
        return string + "\n"
    
def load_dscr(path, dsize=10):
    gg=None
    n=-1
    ds = os.listdir(path)
    len_ds = len(ds)
    for i in ds:#enumerate(["coco17/train/000000000776.jpg.r2d2", "coco17/train/000000000724.jpg.r2d2"]):
        i = path + i
        if not i[-5:] == ".r2d2":
            continue
        if not "landmark" in i:
            continue
        #print(i)
        n += 1
        print("Step: ", n, "/", len_ds)
        #if n > 500:
        #    break
        asd = np.load(i)
        d = asd["descriptors"]
        #d = d[np.random.randint(d.shape[1], size=dsize)]
        #d = d[-dsize:,:]
        #kmeans = KMeans(n_clusters=10)
        #kmeans.fit(d)
        #d = kmeans.cluster_centers_
        if not n:
            gg = d
        else:
            gg = np.concatenate((gg, d))
    return gg

if __name__=="__main__":
    
    
    #path = "coco17/train/"
    #path = "db/"
    if len(sys.argv) < 2:
        exit("Give path to folder where the descriptors are.")
    path = sys.argv[1]
    if not path[-1] == "/":
        path += "/"
    dsize = 10

    """
    kmeans = MiniBatchKMeans(n_clusters=10)
    kmeans.fit(gg)
    print(kmeans.labels_)
    k = 0
    for i in range(10):
        j = gg[kmeans.labels_ == i]
        k += j.shape[0]
        print(j.shape)
    print(k)
    print(kmeans.cluster_centers_.shape)
    np.save("vocab.npy", kmeans.cluster_centers_)
    """
    #gg = load_dscr(path, dsize)
    #asd = Voctree(gg, max_level=2)
    #print(asd)
    #asd.save()
    asd = Voctree(fpath="vocab.npz")
    print(asd)
