from .utils import prospace_pca, transform


class ProspacePCA(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def fit(self, X):
        self.w, self.v, self.A = prospace_pca(X)

    def transform(self, X):
        return transform(X, self.v)[:, : self.dimension]

    def fit_transform(self, X):
        self.w, self.v, self.A = prospace_pca(X)
        return self.transform(X)
