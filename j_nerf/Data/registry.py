class Registry:
    def __init__(self):
        self._modules = {}

    def register_module(self, name=None, module=None):
        def _register_module(module):
            key = name
            if key is None:
                key = module.__name__
            assert key not in self._modules, f"{key} is already registered."
            self._modules[key] = module
            return module

        if module is not None:
            return _register_module(module)

        return _register_module

    def get(self, name):
        assert name in self._modules, f"{name} is not registered."
        return self._modules[name]
