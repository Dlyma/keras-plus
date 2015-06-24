import numpy as np

#             Input:
#               data       - Data matrix. Each row vector of fea is a data point.
#
#     options.ReducedDim   - The dimensionality of the reduced subspace. If 0,
#                         all the dimensions will be kept.
#                         Default is 0.
#
#             Output:
#               eigvector - Each column is an embedding function, for a new
#                           data point (row vector) x,  y = x*eigvector
#                           will be the embedding result of x.
#               eigvalue  - The sorted eigvalue of PCA eigen-problem.

def PCA(data, ReducedDim = 0):
    [nSmp, nFea] = data.shape
    if (ReducedDim > nFea) or (ReducedDim <= 0):
        ReducedDim = nFea
    
    sampleMean = np.mean(data, 0)
    data = (data - np.tile(sampleMean, (nSmp, 1)))
    eigvector, eigvalue, temp = np.linalg.svd(np.dot(data.T,data))
    eigvector = eigvector.T[:ReducedDim].T
    eigvalue = eigvalue[:ReducedDim]
    return eigvector, eigvalue, sampleMean

if __name__ == "__main__":
    data = np.array([[1,2,3,4],[0.2,0.1,3,2],[0.8,0.9,0.4,0.1],[0,0.3,0,2],[1,1,3,3]])
    vec, val, mea = PCA(data, 3)
    print vec, '\n', val
