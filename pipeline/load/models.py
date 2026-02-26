from typing import List, Optional
from datetime import datetime
from sqlalchemy import (String, Integer, DateTime, Enum, Text,
                        ForeignKey, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, declarative_base
from .transform.enums import Chamber, SponsorshipType

Base = declarative_base()

class Representative(Base):
    __tablename__ = "representatives"

    bio_guide_id: Mapped[str] = mapped_column(primary_key = True)
    name: Mapped[str] = mapped_column(String(64), nullable = False)
    party: Mapped[str] = mapped_column(String(16))
    state: Mapped[str] = mapped_column(String(32))
    district: Mapped[int] = mapped_column(Integer)
    chamber: Mapped[Chamber] = mapped_column(Enum(Chamber))

class Bill(Base):
    __tablename__ = "bills"

    id: Mapped[int] = mapped_column(primary_key = True, autoincrement = True)

    congress: Mapped[int] = mapped_column(Integer, primary_key = True)
    type: Mapped[str] = mapped_column(String(8), primary_key = True)
    number: Mapped[int] = mapped_column(Integer, primary_key = True)

    title: Mapped[str] = mapped_column(String(128))    
    chamber: Mapped[Chamber] = mapped_column(Enum(Chamber))
    policy_area: Mapped[str] = mapped_column(String(64))
    summary: Mapped[str] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("congress", "type", "number"),
    )

class BillMembers(Base):
    __tablename__ = "bill_members"

    representative_id: Mapped[str] = mapped_column(ForeignKey("represenatives.bio_guide_id"), primary_key = True)
    bill_id: Mapped[int] = mapped_column(ForeignKey("bills.id"), primary_key = True)
    role: Mapped[SponsorshipType] = mapped_column(Enum(SponsorshipType), nullable = False)


