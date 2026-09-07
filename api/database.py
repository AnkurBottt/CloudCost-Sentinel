from pathlib import Path
# holds all the DB setup code
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DB_DIR = Path("data")
DB_DIR.mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///./data/cloudcost.db"

# connection infra between sqlalchemy and sqlite
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# factory that creates db sessions-(a transaction with the db)- a sqlalchemy object that lets us talk to db 
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# base class(parent class) for all sqlalchemy ORM models/tables inherit from base
class Base(DeclarativeBase):
 pass
...

# FastAPI dependency:
# creates one DB session for a request,
# gives it to the endpoint,
# then closes it afterwards
# allows clean dependcy injection and lifecycle management 
def get_db():
    with SessionLocal() as db:
        yield db

