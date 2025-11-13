from .base_source import DataSource
import importlib

__all__ = [
    "DataSource",
    "PytdxDataSource",
    "TickStore",
    "repository",
    "updater",
    "jobs",
]


def __getattr__(name: str):
    if name == "PytdxDataSource":
        module = importlib.import_module(".pytdx_source", __name__)
        return getattr(module, "PytdxDataSource")
    if name == "TickStore":
        module = importlib.import_module(".tick_store", __name__)
        return getattr(module, "TickStore")
    if name == "repository":
        return importlib.import_module(".repository", __name__)
    if name == "updater":
        return importlib.import_module(".updater", __name__)
    if name == "jobs":
        return importlib.import_module(".jobs", __name__)
    raise AttributeError(name)


if __name__ == "__main__":
    print("data package self-test")
    print("Exports:", __all__)
