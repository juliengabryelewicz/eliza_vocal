import pandas as pd

class Dataset:
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.class_feature = df.columns.values[-1:][0]
        self.input_features = list(df.columns.values[:-1])
        self.unique_labels = pd.factorize(self.labels)[1]

    @classmethod
    def from_csv(cls, file_path: str): return cls(df=pd.read_csv(file_path))

    @property
    def labels(self): return self.data[self.class_feature]

    @property
    def factorized_labels(self): return pd.factorize(self.labels)[0]

    @property
    def without_labels(self): return self.data[self.input_features]

    @property
    def instance_count(self): return len(self.data)