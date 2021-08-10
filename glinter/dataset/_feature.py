class DimerFeature:
    def __init__(self, s):
        features = s.split(',')
        if len(features) == 0:
            raise ValueError('empty feature list')
        # check used features
        for k in features:
            if k not in self.groups:
                raise ValueError(f'{k} is not defined')

        if 'esm' in features and 'pickled-esm' in features:
            raise ValueError(f'use one of "esm" or "pickled-esm"')
        self.features = features

    @property
    def groups(self):
        return set([
            'ccm', 'esm', 'pickled-esm', 'ca-embed',
            'coordinate-ca-graph', 'distance-ca-graph',
            'atom-graph', 'surface-graph', 'heavy-atom-graph',
        ])

    def use(self, *keys):
        for key in keys:
            if key in self:
                return True
        return False

    def __contains__(self, key):
        if key not in self.groups:
            raise ValueError(f'{key} is not a defined feature group')
        return key in self.features

    def __repr__(self):
        return ','.join(self.features)
