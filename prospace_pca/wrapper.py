from .utils import prospace_pca, reduce_dimensions


class ProspacePCA(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def fit(self, X):
        self.w, self.v, self.A = prospace_pca(X)

    def transform(self, X):
        return reduce_dimensions(X, self.v, self.dimension)

    def fit_transform(self, X):
        self.w, self.v, self.A = prospace_pca(X)
        return self.transform(X)
