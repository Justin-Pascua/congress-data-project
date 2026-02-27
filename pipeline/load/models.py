from typing import List, Optional
from datetime import datetime
from sqlalchemy import (String, Integer, DateTime, Enum, Text, Boolean,
                        UniqueConstraint, ForeignKey, ForeignKeyConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base
from ..transform.enums import Chamber, SponsorshipType, BillType

MAX_STR_LEN = 40

Base = declarative_base()

class Member(Base):
    __tablename__ = "members"

    bio_guide_id: Mapped[str] = mapped_column(String(16), primary_key = True)
    name: Mapped[str] = mapped_column(String(64))
    party: Mapped[str] = mapped_column(String(16), nullable = True)
    state: Mapped[str] = mapped_column(String(32))
    district: Mapped[int] = mapped_column(Integer, nullable = True)
    chamber: Mapped[Chamber] = mapped_column(Enum(Chamber))

    def __repr__(self):
        """Generate string representation dynamically."""
        model_name = self.__class__.__name__
        
        # Get primary key column(s) first
        pk = ', '.join(
            f"{col.name}={getattr(self, col.name)}" 
            for col in self.__table__.primary_key.columns
        )

        other_cols = []
        for col in self.__table__.columns:
            if not col.primary_key: 
                value = getattr(self, col.name)
                
                # Truncate long strings
                if isinstance(value, str) and len(value) > MAX_STR_LEN:
                    value = value[:MAX_STR_LEN-3] + '...'
                other_cols.append(f"{col.name}='{value}'")
        
        all_attrs = [pk] + other_cols
        return f"<{model_name}({', '.join(all_attrs)})>"

class Bill(Base):
    __tablename__ = "bills"

    congress_num: Mapped[int] = mapped_column(Integer, primary_key = True)
    bill_type: Mapped[BillType] = mapped_column(Enum(BillType), primary_key = True)
    bill_num: Mapped[int] = mapped_column(Integer, primary_key = True)

    title: Mapped[str] = mapped_column(Text)    
    chamber: Mapped[Chamber] = mapped_column(Enum(Chamber))
    policy_area: Mapped[str] = mapped_column(String(64), nullable = True)
    summary: Mapped[str] = mapped_column(Text, nullable = True)

    def __repr__(self):
        """Generate string representation dynamically."""
        model_name = self.__class__.__name__
        
        # Get primary key column(s) first
        pk = ', '.join(
            f"{col.name}={getattr(self, col.name)}" 
            for col in self.__table__.primary_key.columns
        )

        other_cols = []
        for col in self.__table__.columns:
            if not col.primary_key: 
                value = getattr(self, col.name)
                
                # Truncate long strings
                if isinstance(value, str) and len(value) > MAX_STR_LEN:
                    value = value[:MAX_STR_LEN-3] + '...'
                other_cols.append(f"{col.name}='{value}'")
        
        all_attrs = [pk] + other_cols
        return f"<{model_name}({', '.join(all_attrs)})>"

class BillSponsorship(Base):
    __tablename__ = "bill_sponsorship"

    bio_guide_id: Mapped[str] = mapped_column(ForeignKey("members.bio_guide_id", ondelete = "CASCADE"), primary_key = True)
    
    congress_num: Mapped[int] = mapped_column(Integer, primary_key = True)
    bill_type: Mapped[BillType] = mapped_column(Enum(BillType), primary_key = True)
    bill_num: Mapped[int] = mapped_column(Integer, primary_key = True)
    
    sponsorship_type: Mapped[SponsorshipType] = mapped_column(Enum(SponsorshipType), nullable = False)
    is_active: Mapped[bool] = mapped_column(Boolean, default = True)
    last_refresh: Mapped[datetime] = mapped_column(DateTime, default = lambda: datetime.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ['congress_num', 'bill_type', 'bill_num'],
            ['bills.congress_num', 'bills.bill_type', 'bills.bill_num'],
            ondelete = "CASCADE"
        ),
    )

    def __repr__(self):
        """Generate string representation dynamically."""
        model_name = self.__class__.__name__
        
        # Get primary key column(s) first
        pk = ', '.join(
            f"{col.name}={getattr(self, col.name)}" 
            for col in self.__table__.primary_key.columns
        )

        other_cols = []
        for col in self.__table__.columns:
            if not col.primary_key: 
                value = getattr(self, col.name)
                
                # Truncate long strings
                if isinstance(value, str) and len(value) > MAX_STR_LEN:
                    value = value[:MAX_STR_LEN-3] + '...'
                other_cols.append(f"{col.name}='{value}'")
        
        all_attrs = [pk] + other_cols
        return f"<{model_name}({', '.join(all_attrs)})>"
    
