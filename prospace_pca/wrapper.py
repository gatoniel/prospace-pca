from .utils import prospace_pca, transform, normalize


class ProspacePCA(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def fit(self, X):
        self.w, self.v, self.A = prospace_pca(X)

    def transform(self, X):
        return transform(normalize(X), self.v)[:, : self.dimension]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
