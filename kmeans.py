UBIT = 'emmanueljohnson'
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
np.random.seed(sum([ord(c) for c in UBIT]))

#Number of Clusters
K = 3

#Input Points
PTS = np.array([[5.9, 3.2],
              [4.6, 2.9],
              [6.2, 2.8],
              [4.7, 3.2],
              [5.5, 4.2],
              [5.0, 3.0],
              [4.9, 3.1],
              [6.7, 3.1],
              [5.1, 3.8],
              [6.0, 3.0]])

#Initial Centroids
CENTROIDS = np.array([[6.2, 3.2],
                      [6.6, 3.7],
                      [6.5, 3.0]])

#Colors List
COLORS = ['red','green','blue']

#Read the image using opencv
def get_image(path):
    return cv2.imread(path)

#Read the image in gray scale using opencv
def get_image_gray(path):
    return cv2.imread(path,0)

#Show the resulting image
def show_image(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Save the resulting image
def save_image(name,image):
    cv2.imwrite(name,image) 

#Plot the points and its values if text is true
def plot_points(points,clusters,marker,text=True):
    for i,x in enumerate(points):
        plt.scatter(x[0], x[1], edgecolor=COLORS[clusters[i]], facecolor='white', linewidth='1', marker=marker)
        if text:
            plt.text(x[0]+0.03, x[1]-0.05, "("+str(x[0])+","+str(x[1])+")", color=COLORS[clusters[i]], fontsize='small')

#Plot the centroids and its values if text is true
def plot_centroids(centroids,colors,marker,text=True):
    for c,color in zip(centroids,colors):
        plt.scatter(c[0], c[1], marker=marker, s=200, c=color)
        if text:
            plt.text(c[0]+0.03, c[1]-0.05, "("+str(c[0])+","+str(c[1])+")", color=color, fontsize='small')

#Find Euclidean distance between two points using numpy
def euclidean_dist(pt1, pt2):
    return np.linalg.norm(pt1 - pt2, axis=1)

#Find the minimum distance between a point and the centroids
#Returns the cluster index in which the point belongs
def get_min_dist_index(distances):
    minDist = distances[0]
    for d in distances:
        if minDist > d:
            minDist = d
    return distances.index(minDist)

#Get the classification vector
def get_clusters(points,centroids,r):
    clusters = np.zeros(r, dtype=int)
    for i in range(r):
        allDistances = euclidean_dist(points[i], centroids)
        clusters[i] = get_min_dist_index(allDistances.tolist())
    return clusters

#Compute the new centroids
def get_new_centroids(points,centroids,clusters,r):
    for i in range(K):
        pts = [points[j] for j in range(r) if clusters[j] == i]
        centroids[i] = np.mean(pts, axis=0)
    return centroids

#kmeans implementation
def kmeans(centroids,points,maxIterations,nolimit):
    prev_centroids = np.zeros(centroids.shape,dtype=int)
    plen = len(points)
    clusters = np.zeros(plen)
    i = 0
    iters_taken = -1
    while i<maxIterations:
        iters_taken += 1
        clusters = get_clusters(points,centroids,plen)
        prev_centroids = np.copy(centroids)
        centroids = get_new_centroids(points,centroids,clusters,plen)
        difference = euclidean_dist(prev_centroids,centroids)
        #Check if previous and new centroids are same
        #If same they are converged
        if np.count_nonzero(difference) == 0:
            print('Converged in '+str(iters_taken)+" iterations")
            return centroids,clusters
        #Run until a particular iteration
        if not nolimit:
            i+=1
    return centroids,clusters

#Get random indexes between a specified range
def get_random_index(r,n):
    rPicks = np.random.randint(0, r, n)
    #rPicks = random.sample(range(r), n)
    return rPicks

#Image Quantization implementation
def image_quantization():
    k_clusters = [3, 5, 10, 20]
    for kc in k_clusters:
        K = kc
        image = get_image('baboon.jpg')
        orgH, orgW = image.shape[0], image.shape[1]
        image = image.reshape((orgH * orgW, 3))

        #Choose some random points from the image as
        #an initial centroids
        rIndexes = get_random_index(len(image), K)
        icentroids = []
        for r in rIndexes:
            icentroids.append(image[r].tolist())
        icentroids = np.array(icentroids)

        print('Finding Centroids and Clusters for K = '+str(kc))
        print('__This process might take few minutes__')
        centroids, clusters = kmeans(icentroids, image, 1, True)

        #Replace the image values with the value of the centroid
        #depending on its cluster
        for i in range(len(clusters)):
            image[i] = centroids[clusters[i]]

        image = image.reshape((orgH, orgW, 3))

        print('Image Quantization done for K = '+str(kc))
        save_image("task3_baboon_"+str(kc)+".jpg", image)
        print('\n')

#References
#https://brilliant.org/wiki/gaussian-mixture-model/
#https://www.mathworks.com/help/stats/clustering-using-gaussian-mixture-models.html

#Finding the probability distribution function value
def gmm_prob_dist(inputPt, mean, cov):
    return multivariate_normal.pdf(inputPt, mean, cov)

#Finding the log of probability distribution function
def gmm_prob_dist_log(inputPt, mean, cov):
    return multivariate_normal.logpdf(inputPt, mean, cov)

#Finding the probability of the points belonging to a 
#particular cluster
def gmm_probability(pts, pis, mus, sigmas):
    prob = np.zeros((len(pts), K))
    for x in range(len(pts)):
        for y in range(K):
            N = gmm_prob_dist(pts[x], mus[y], sigmas[y])
            prob[x, y] = pis[y]*N
        prob_sum = np.sum(prob[x])
        prob[x] = prob[x]/prob_sum
    return prob

#Update the pi values
def gmm_updatePis(gp):
    N = len(gp)
    gp_sum = np.sum(gp, axis=0)
    return gp_sum/N

#Update the mean values
def gmm_updateMus(gp,pts):
    numer = list()
    numerValue = zip(np.sum(gp*pts[:, 0][:, np.newaxis], axis=0),
                    np.sum(gp*pts[:, 1][:, np.newaxis], axis=0))
    for n in numerValue:
        numer.append(n)
    numer = np.array(numer)
    denom = np.sum(gp, axis=0)[:, np.newaxis]
    return numer/denom

#Update the covariance values
def gmm_updateSigmas(gp, pts, mus):
    gpLen = len(gp)
    numer = np.zeros([gpLen, K, 2, 2])
    for i in range(gpLen):
        for j in range(K):
            xMinMu = (pts[i]-mus[j])
            xMinMuSq = np.dot(xMinMu[:, np.newaxis], xMinMu[np.newaxis, :])
            numer[i][j] = gp[i][j] * xMinMuSq
    numerSum = np.sum(numer, axis=0)
    denom = np.sum(gp, axis=0)[:, np.newaxis, np.newaxis]
    return numerSum/denom

#References
#https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

#plot the graph for the given points
def plot_graph(dataset, prob, sigmas, mus, iteration):
    for i, (x, p) in enumerate(zip(dataset, prob)):
        pindex = np.argmax(p)
        plt.scatter(x[0], x[1], edgecolors=COLORS[pindex],
                    facecolor='white', linewidth='1', marker='.')

    for i in range(3):
        plot_cov_ellipse(sigmas[i], mus[i], nstd=2, alpha=0.5, color=COLORS[i])

    plt.savefig('task3_gmm_iter'+str(iteration)+'.jpg')
    plt.clf()

def gmm(pts, mus, sigmas, pis, maxIterations, nolimit, plot=False):
    #Initialization step
    i = 0
    iters_taken = -1
    # gp = gmm_probability(pts, pis, mus, sigmas)
    # prev_loss = log_likelihood(gp, pts, pis, mus, sigmas)
    while i < maxIterations:
        iters_taken += 1
        
        gp = gmm_probability(pts, pis, mus, sigmas)
        
        #Plotting the initial graph
        if iters_taken == 0:
            plot_graph(pts, gp, sigmas, mus, iters_taken)

        #update the pi, mu and sigma
        updatedPis = gmm_updatePis(gp)
        updatedMus = gmm_updateMus(gp, pts)
        updatedSigmas = gmm_updateSigmas(gp, pts, updatedMus)
        
        #Update the value of pi, mean and covariance
        pis = updatedPis
        mus = updatedMus
        sigmas = updatedSigmas
        
        if plot == True:
            plot_graph(pts, gp, sigmas, mus, iters_taken+1)
            print('\nUpdated Mu after Iteration '+str(iters_taken+1))
            print(mus)
        if not nolimit:
            i += 1
    
    return gp, mus, sigmas, pis

def task_gmm():
    
    sigmas = np.array([[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]])
    pis = np.array([1/3, 1/3, 1/3])
    
    rprob, rmu, rsigma, rpi = gmm(PTS, CENTROIDS, sigmas, pis, 1, False)

    print('\nTask 3.5 (a)')
    print('Updated Mu')
    print(rmu)

    #Old Faithful Dataset
    print('\nTask 3.5 (b)')
    old_faithful = np.array([[3.600, 79], [1.800, 54], [3.333, 74], [2.283, 62],[4.533, 85], [2.883, 55], [4.700, 88], [3.600, 85], [1.950, 51], [4.350, 85],
                            [1.833, 54], [3.917, 84], [4.200, 78], [1.750, 47], [4.700, 83], [2.167, 52], [1.750, 62], [4.800, 84], [1.600, 52], [4.250, 79],
                            [1.800, 51], [1.750, 47], [3.450, 78], [3.067, 69], [4.533, 74], [3.600, 83], [1.967, 55], [4.083, 76], [3.850, 78], [4.433, 79],
                            [4.300, 73], [4.467, 77], [3.367, 66], [4.033, 80], [3.833, 74], [2.017, 52], [1.867, 48], [4.833, 80], [1.833, 59], [4.783, 90],
                            [4.350, 80], [1.883, 58], [4.567, 84], [1.750, 58], [4.533, 73], [3.317, 83], [3.833, 64], [2.100, 53], [4.633, 82], [2.000, 59],
                            [4.800, 75], [4.716, 90], [1.833, 54], [4.833, 80], [1.733, 54], [4.883, 83], [3.717, 71], [1.667, 64], [4.567, 77], [4.317, 81],
                            [2.233, 59], [4.500, 84], [1.750, 48], [4.800, 82], [1.817, 60], [4.400, 92], [4.167, 78], [4.700, 78], [2.067, 65], [4.700, 73],
                            [4.033, 82], [1.967, 56], [4.500, 79], [4.000, 71], [1.983, 62], [5.067, 76], [2.017, 60], [4.567, 78], [3.883, 76], [3.600, 83],
                            [4.133, 75], [4.333, 82], [4.100, 70], [2.633, 65], [4.067, 73], [4.933, 88], [3.950, 76], [4.517, 80], [2.167, 48], [4.000, 86],
                            [2.200, 60], [4.333, 90], [1.867, 50], [4.817, 78], [1.833, 63], [4.300, 72], [4.667, 84], [3.750, 75], [1.867, 51], [4.900, 82],
                            [2.483, 62], [4.367, 88], [2.100, 49], [4.500, 83], [4.050, 81], [1.867, 47], [4.700, 84], [1.783, 52], [4.850, 86], [3.683, 81],
                            [4.733, 75], [2.300, 59], [4.900, 89], [4.417, 79], [1.700, 59], [4.633, 81], [2.317, 50], [4.600, 85], [1.817, 59], [4.417, 87],
                            [2.617, 53], [4.067, 69], [4.250, 77], [1.967, 56], [4.600, 88], [3.767, 81], [1.917, 45], [4.500, 82], [2.267, 55], [4.650, 90],
                            [1.867, 45], [4.167, 83], [2.800, 56], [4.333, 89], [1.833, 46], [4.383, 82], [1.883, 51], [4.933, 86], [2.033, 53], [3.733, 79],
                            [4.233, 81], [2.233, 60], [4.533, 82], [4.817, 77], [4.333, 76], [1.983, 59], [4.633, 80], [2.017, 49], [5.100, 96], [1.800, 53],
                            [5.033, 77], [4.000, 77], [2.400, 65], [4.600, 81], [3.567, 71], [4.000, 70], [4.500, 81], [4.083, 93], [1.800, 53], [3.967, 89],
                            [2.200, 45], [4.150, 86], [2.000, 58], [3.833, 78], [3.500, 66], [4.583, 76], [2.367, 63], [5.000, 88], [1.933, 52], [4.617, 93],
                            [1.917, 49], [2.083, 57], [4.583, 77], [3.333, 68], [4.167, 81], [4.333, 81], [4.500, 73], [2.417, 50], [4.000, 85], [4.167, 74],
                            [1.883, 55], [4.583, 77], [4.250, 83], [3.767, 83], [2.033, 51], [4.433, 78], [4.083, 84], [1.833, 46], [4.417, 83], [2.183, 55],
                            [4.800, 81], [1.833, 57], [4.800, 76], [4.100, 84], [3.966, 77], [4.233, 81], [3.500, 87], [4.366, 77], [2.250, 51], [4.667, 78],
                            [2.100, 60], [4.350, 82], [4.133, 91], [1.867, 53], [4.600, 78], [1.783, 46], [4.367, 77], [3.850, 84], [1.933, 49], [4.500, 83],
                            [2.383, 71], [4.700, 80], [1.867, 49], [3.833, 75], [3.417, 64], [4.233, 76], [2.400, 53], [4.800, 94], [2.000, 55], [4.150, 76],
                            [1.867, 50], [4.267, 82], [1.750, 54], [4.483, 75], [4.000, 78], [4.117, 79], [4.083, 78], [4.267, 78], [3.917, 70], [4.550, 79],
                            [4.083, 70], [2.417, 54], [4.183, 86], [2.217, 50], [4.450, 90], [1.883, 54], [1.850, 54], [4.283, 77], [3.950, 79], [2.333, 64],
                            [4.150, 75], [2.350, 47], [4.933, 86], [2.900, 63], [4.583, 85], [3.833, 82], [2.083, 57], [4.367, 82], [2.133, 67], [4.350, 74],
                            [2.200, 54], [4.450, 83], [3.567, 73], [4.500, 73], [4.150, 88], [3.817, 80], [3.917, 71], [4.450, 83], [2.000, 56], [4.283, 79],
                            [4.767, 78], [4.533, 84], [1.850, 58], [4.250, 83], [1.983, 43], [2.250, 60], [4.750, 75], [4.117, 81], [2.150, 46], [4.417, 90],
                            [1.817, 46], [4.467, 74]])
    
    omus = np.array([[4.0, 81],
                       [2.0, 57],
                       [4.0, 71]])

    osigmas = np.array([[[1.30, 13.98], [13.98, 184.82]], 
                        [[1.30, 13.98], [13.98, 184.82]], 
                        [[1.30, 13.98], [13.98, 184.82]]])

    rprob, rmu, rsigma, rpi = gmm(old_faithful, omus, osigmas, pis, 5, False, plot=True)

def main():

    centroids,clusters = kmeans(CENTROIDS,PTS,1,False)
    
    print('Task 3.1 Classification Vector')
    print(clusters)

    print('\nTask 3.1 Plotting Classification')
    plot_points(PTS,clusters,'^')
    plt.savefig('task3_iter1_a.jpg')

    print('\nTask 3.2 Updated Mu (Centroids)')
    print(centroids)

    print('\nTask 3.2 Plotting Updated Mu (Centroids)')
    plot_centroids(np.round(centroids, decimals=2), COLORS, '.')
    plt.savefig('task3_iter1_b.jpg')
    plt.clf()

    centroids,clusters = kmeans(CENTROIDS,PTS,2,False)

    print('\nTask 3.3 Classification Vector')
    print(clusters)

    print('\nTask 3.3 Updated Mu (Centroids)')
    print(centroids)

    print('\nTask 3.3 Plotting Classification and Updated Mu (Centroids)')
    plot_points(PTS, clusters, '^')
    plt.savefig('task3_iter2_a.jpg')
    plot_centroids(np.round(centroids, decimals=2), COLORS, '.')
    plt.savefig('task3_iter2_b.jpg')
    plt.clf()
    
    print('Task 3.4')
    image_quantization()

    print('Task 3.5')
    task_gmm()

if __name__ == '__main__':
    main()
