import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

class DummyQuery:
    def filter(self, *args, **kwargs):
        return self

    def delete(self, synchronize_session=False):
        return 0


class DummySession:
    def query(self, *args, **kwargs):
        return DummyQuery()

    def add(self, obj):
        return None

    def execute(self, *args, **kwargs):
        class R:
            def scalars(self):
                class S:
                    def all(self):
                        return []

                return S()

            def scalar_one_or_none(self):
                return None

        return R()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def bulk_save_objects(self, objs):
        return None

    def get_bind(self):
        return None


_engine = None

def _get_real_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, future=True)
    return _engine

def get_session():
    engine = _get_real_engine()
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return Session()

def get_engine():
    return _get_real_engine()
