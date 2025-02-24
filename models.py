from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
engine = create_engine('sqlite:///knowledge_base.db')
Session = sessionmaker(bind=engine)

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text)

class DocumentMeta(Base):
    __tablename__ = 'document_meta'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    summary = Column(String(512))
    created_at = Column(DateTime, default=datetime.now)

# 创建数据库表
Base.metadata.create_all(engine) 