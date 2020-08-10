# coding: utf-8
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_db_engine():
    return create_engine('sqlite:///' + os.path.join(BASE_DIR, 'db.sqlite'),
                         echo=False,
                         pool_recycle=3600,
                         poolclass=NullPool,
                         connect_args={'check_same_thread': False}
                         )


def get_db_session(expire_on_commit=True, autocommit=False):
    engine = get_db_engine()
    Session = sessionmaker(bind=engine, expire_on_commit=expire_on_commit, autocommit=autocommit)
    session = Session()
    return session
