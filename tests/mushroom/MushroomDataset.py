import numpy as np
from lib.BinaryDataset import BinaryDataset

attribute_map = [
    ["b", "c", "x", "f", "k", "s"],  # cap-shape:                bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
    ["f", "g", "y", "s"],  # cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
    ["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"],  # cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
    ["t", "f"],  # bruises?:                 bruises=t,no=f
    ["a", "l", "c", "y", "f", "m", "n", "p", "s"],  # odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
    ["a", "d", "f", "n"],  # gill-attachment:          attached=a,descending=d,free=f,notched=n
    ["c", "w", "d"],  # gill-spacing:             close=c,crowded=w,distant=d
    ["b", "n"],  # gill-size:                broad=b,narrow=n
    ["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"],  # gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
    ["e", "t"],  # stalk-shape:              enlarging=e,tapering=t
    ["b", "c", "u", "e", "z", "r", "?"],  # stalk-root:               bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
    ["f", "y", "k", "s"],  # stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    ["f", "y", "k", "s"],  # stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    ["n", "b", "c", "g", "o", "p", "e", "w", "y"],  # stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
    ["n", "b", "c", "g", "o", "p", "e", "w", "y"],  # stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
    ["p", "u"],  # veil-type:                partial=p,universal=u
    ["n", "o", "w", "y"],  # veil-color:               brown=n,orange=o,white=w,yellow=y
    ["n", "o", "t"],  # ring-number:              none=n,one=o,two=t
    ["c", "e", "f", "l", "n", "p", "s", "z"],  # ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
    ["k", "n", "b", "h", "r", "o", "u", "w", "y"],  # spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
    ["a", "c", "n", "s", "v", "y"],  # population:               abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
    ["g", "l", "m", "p", "u", "w", "d"],  # habitat:                  grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
]


def parse_mushroom_dataset(raw_dataset_path):
    # Covert raw Mushroom data into a workable numpy matrix
    dataset = []

    with open(raw_dataset_path) as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.replace("\n", "")
            attributes = line.split(",")[1:]
            row = np.array([], dtype=bool)
            row = []
            for id, attribute in enumerate(attributes):
                columns = np.zeros(len(attribute_map[id]), dtype=bool)
                columns[attribute_map[id].index(attribute)] = True
                row = np.concatenate((row, columns))
            dataset.append(row)

    return BinaryDataset(np.array(dataset, dtype=bool))
