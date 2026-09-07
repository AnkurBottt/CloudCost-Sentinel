from sqlalchemy import Integer, String, Float
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base

class Prediction(Base): # basically an python orm class mapped to the actual predictions sql table
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    provider: Mapped[str] = mapped_column(String(10), nullable=False)
    service: Mapped[str] = mapped_column(String(10), nullable=False)
    region: Mapped[str] = mapped_column(String(20), nullable=False)
    daily_cost: Mapped[float] = mapped_column(Float, nullable=False)
    usage_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    resource_count: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[int] = mapped_column(Integer, nullable=False)
...


