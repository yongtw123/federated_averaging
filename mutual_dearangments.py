import numpy as np

np.random.seed(1)

def mutder10_pick10():
    """This runs for a long time"""
    bo=[]

    while len(bo) != 10:
        perm = np.random.permutation(10)
        if not any([any([i==j for i,j in zip(perm, b)]) for b in bo]):
            bo.append(perm)

    return [p.tolist() for p in bo]

def mutder10_pick5():
    bo=[]

    while len(bo) != 10:
        perm = np.random.permutation(10)
        p = perm[0:5]
        q = np.copy(perm[5:])
        if not any([any([i==j for i,j in zip(p, b)]) for b in bo]):
            bo.append(p)
            while any([any([i==j for i,j in zip(q, b)]) for b in bo]):
                np.random.shuffle(q)
            bo.append(q)

    return [p.tolist() for p in bo]
    
def mutder10_pick2():
    perm = np.random.permutation(10)
    p = np.copy(perm[0:5])
    q = np.copy(perm[5:])
    np.random.shuffle(p)
    np.random.shuffle(q)
    qq = np.copy(q)
    while any([i==j for i,j in zip(q, qq)]):
        np.random.shuffle(qq)
    bo = np.column_stack((
        np.concatenate((perm[0:5], qq)),
        np.concatenate((perm[5:], p))
    ))
    
    return bo
    
print(mutder10_pick2())