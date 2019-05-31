import numpy as np

# Some elementary functions to speak the same language as the paper
# (at some point we'll just replace the occurrence of the calls with the function body itself)
def push(x,stack):
    stack.append(x)

def pop(stack):
    return stack.pop()

def top(stack):
    return stack[-1]

def nextToTop(stack):
    return stack[-2]


# perhaps inefficient but clear implementation
def nonleftTurn(a,b,c):   
    
    d1 = b-a
    d2 = c-b

    return np.cross(d1,d2)<=0

def nonrightTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)>=0


def slope(a,b):
    ax,ay = a
    bx,by = b
    return (by-ay)/(bx-ax)

def notBelow(t,p1,p2):
    p1x,p1y = p1
    p2x,p2y = p2
    tx,ty = t
    m = (p2y-p1y)/(p2x-p1x)
    b = (p2x*p1y - p1x*p2y)/(p2x-p1x)
    return (ty >= tx*m+b)

kPrime = None

# Because we cannot have negative indices in Python (they have another meaning), I use a dictionary

def algorithm1(P):
    global kPrime

    S = []
    P[-1] = np.array((-1,-1))

    push(P[-1],S)
    push(P[0],S)

    for i in range(1,kPrime+1):
        while len(S)>1 and nonleftTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)    
    return S

def algorithm2(P,S):
    global kPrime

    print("")
    print("######### algorithm2")
    print("")
    print("")
    
    Sprime = S[::-1]     # reverse the stack

    F1 = np.zeros((kPrime+1,))
    for i in range(1,kPrime+1):
        F1[i] = slope(top(Sprime),nextToTop(Sprime))
        print(" ##F1 ===> " , F1)
        P[i-1] = P[i-2]+P[i]-P[i-1]
        if notBelow(P[i-1],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonleftTurn(P[i-1],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i-1],Sprime)

    print(" #F1 ===> " , F1)
    return F1

def algorithm3(P):
    global kPrime
    
    P[kPrime+1] = P[kPrime]+np.array((1.0,0.0))

    S = []
    push(P[kPrime+1],S)
    push(P[kPrime],S)
    for i in range(kPrime-1,0-1,-1):  # k'-1,k'-2,...,0
        while len(S)>1 and nonrightTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S

def algorithm4(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack
    
    F0 = np.zeros((kPrime+1,))
    for i in range(kPrime,1-1,-1):   # k',k'-1,...,1
        F0[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i] = P[i-1]+P[i+1]-P[i]
        if notBelow(P[i],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonrightTurn(P[i],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i],Sprime)
    return F0[1:]

def prepareData(calibrPoints):
    global kPrime

    ptsSorted = sorted(calibrPoints)
    
    xs = np.fromiter((p[0] for p in ptsSorted),float)
    ys = np.fromiter((p[1] for p in ptsSorted),float)
    ptsUnique,ptsIndex,ptsInverse,ptsCounts = np.unique(xs, 
                                                        return_index=True,
                                                        return_counts=True,
                                                        return_inverse=True)

    print("xs =======> ",xs)
    print("ys =======> ",ys)
    print("ptsInverse =======> ",ptsInverse)

    a = np.zeros(ptsUnique.shape)

    np.add.at(a,ptsInverse,ys)
    print("np.add.at(a,ptsInverse,ys) =======> ",a)

    # now a contains the sums of ys for each unique value of the objects
    
    w = ptsCounts
    print("a = np.zeros(ptsUnique.shape) =======> ",a)
    print("w = ptsCounts =======> ",w)
    yPrime = a/w
    print("yPrime =======> ",yPrime)
    yCsd = np.cumsum(w*yPrime)   # Might as well do just np.cumsum(a)
    xPrime = np.cumsum(w)
    kPrime = len(xPrime)
    
    return yPrime,yCsd,xPrime,ptsUnique

def computeF(xPrime,yCsd):    
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})
    print("P =======> ",P)
    print("zip(xPrime,yCsd) =======> ",enumerate(zip(xPrime,yCsd)))

    
    S = algorithm1(P)

    F1 = algorithm2(P,S)

    
    # P = {}
    # P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})    
    
    S = algorithm3(P)
    F0 = algorithm4(P,S)
    
    return F0,F1

def getFVal(F0,F1,ptsUnique,testObjects):
    pos0 = np.searchsorted(ptsUnique[1:],testObjects,side='right')
    print('ptsUnique[1:] ===== > ' ,ptsUnique[1:])
    pos1 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    print('ptsUnique[:-1] ===== > ' ,ptsUnique[:-1])
    
    return F0[pos0],F1[pos1]

def ScoresToMultiProbs(calibrPoints,testObjects):
    # sort the points, transform into unique objects, with weights and updated values
    yPrime,yCsd,xPrime,ptsUnique = prepareData(calibrPoints)
    
    # print("")
    # print(" xPrime ===> " , xPrime)
    # print(" yCsd ===> " , yCsd)
    
    # compute the F0 and F1 functions from the CSD
    F0,F1 = computeF(xPrime,yCsd)
    
    # compute the values for the given test objects
    p0,p1 = getFVal(F0,F1,ptsUnique,testObjects)
                    
    return p0,p1

def computeF1(yCsd,xPrime):
    global kPrime
    
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})
    
    S = algorithm1(P)
    F1 = algorithm2(P,S)
    
    return F1

def ScoresToMultiProbsV2(calibrPoints,testObjects):
    # sort the points, transform into unique objects, with weights and updated values
    yPrime,yCsd,xPrime,ptsUnique = prepareData(calibrPoints)
   
    # compute the F0 and F1 functions from the CSD
    F1 = computeF1(yCsd,xPrime)
    pos1 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    p1 = F1[pos1]
    
    yPrime,yCsd,xPrime,ptsUnique = prepareData((-x,1-y) for x,y in calibrPoints)    
    F0 = 1 - computeF1(yCsd,xPrime)
    pos0 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    p0 = F0[pos0]
    
    return p0,p1






li = [(0.5,0),(0.2,0),(0.7,1),(0.6,1),(0.8,1),(0.9,1)]
ScoresToMultiProbs(li,[0.4,0.6])



# yPrime,yCsd,xPrime,ptsUnique = prepareData(li)

# print("1 =======> ",yPrime)
# print("2 =======> ",yCsd)
# print("3 =======> ",xPrime)
# print("4 =======> ",ptsUnique)
# print("5 =======> ",li)
# print("6 =======> ",sorted(li))

# a = np.zeros(3)
# a = np.array([0, 2, 3, 3, 4])
# # p = np.array([0, 2, 3, 3, 4])
# y = np.array([1, 1, 1, 1, 3])

# # np.add.at(a,p,y)

# print(a/y)
# print(p)
# print(y)

# a = np.array([1 ,1 ,1 ,1 ,1])
# p = np.array([0, 2, 3, 3, 4])
# y = np.array([1, 0, 1, 0, 3])

# for i in p :
#     a[i] += y[i]

# print(a)
# print(p)
# print(y)