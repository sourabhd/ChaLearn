# Class to build visual vocabulary
#
# References:
# 1. http://www.slideshare.net/mrwalle/apa-pycon-2012-machine-learning-for-computer-vision-applications
# 2. http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
# 3. http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
# 4. http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans 

import scipy as sp
import numpy as np
import pylab as pl
from sklearn import cluster
from sklearn import metrics
from time import time
import sys



class BOW(object):

    def vq(self, data, voc_size, gt_labels):
        self.data = data
        self.voc_size = voc_size
        self.init = 'k-means++'
        self.gt_labels = gt_labels
        self.run_kmeans()
        #self.vector_quantize()

    def run_kmeans(self):
        np.set_printoptions(threshold='nan')
        #print self.data
        self.kmeans = cluster.KMeans(init=self.init, n_clusters=self.voc_size, n_init=10, precompute_distances=True)

#        self.bench_k_means()

        self.kmeans.fit(self.data)
        self.pred_labels = self.kmeans.predict(self.data)
        self.centroids = self.kmeans.cluster_centers_
        #print
        #print
        #print self.pred_labels  
        #print self.centroids

        # Code from sklearn docs for visualization in case of 2-D data
        # Uncomment to visualize
        # Step size of the mesh. Decrease to increase the quality of the VQ.
#        h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
#
#        # Plot the decision boundary. For that, we will assign a color to each
#        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
#        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
#        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
#        # Obtain labels for each point in mesh. Use last trained model.
#        Z = self.kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
#
#        # Put the result into a color plot
#        Z = Z.reshape(xx.shape)
#        pl.figure(1)
#        pl.clf()
#        pl.imshow(Z, interpolation='nearest',
#                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#                cmap=pl.cm.Paired,
#                aspect='auto', origin='lower')
#
#        pl.plot(self.data[:, 0], self.data[:, 1], 'k.', markersize=10)
#
#        # Plot the centroids as a white X
#        centroids = self.kmeans.cluster_centers_
#        pl.scatter(centroids[:, 0], centroids[:, 1],
#                marker='x', linewidths=3, s=169,
#                color='w', zorder=10)
#        pl.title('K-means clustering\n'
#                'Centroids are marked with white cross')
#        pl.xlim(x_min, x_max)
#        pl.ylim(y_min, y_max)
#        pl.xticks(())
#        pl.yticks(())
#        pl.show()

    def bench_k_means(self):
        estimator = self.kmeans
        t0 = time()
        estimator.fit(self.data)
        print("n_clusters: %d, \t n_samples %d, \t n_features %d" % (self.voc_size, self.data.shape[0], self.data.shape[1]))
        print(79 * '_')
        print('% 9s' % 'init' '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
        print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (str(self.init), (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(self.gt_labels, estimator.labels_),
             metrics.completeness_score(self.gt_labels, estimator.labels_),
             metrics.v_measure_score(self.gt_labels, estimator.labels_),
             metrics.adjusted_rand_score(self.gt_labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(self.gt_labels,  estimator.labels_),
             metrics.silhouette_score(self.data, estimator.labels_,
                                      metric='euclidean',
                                    )))


    def calc_bow_representation(self,fv):
        self.bow = sp.spatial.distance.cdist(fv,self.centroids,'euclidean')
        print self.bow.shape

