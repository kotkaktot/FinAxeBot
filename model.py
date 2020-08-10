# coding: utf-8
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from connections import get_db_engine
import datetime

Base = declarative_base()
metadata = Base.metadata


class Queries(Base):
    __tablename__ = 'Queries'

    user_id = Column(Integer)
    first_name = Column(String)
    last_name = Column(String)
    username = Column(String)
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, primary_key=True)


# Uncomment only for db creation
engine = get_db_engine()
metadata.create_all(engine)