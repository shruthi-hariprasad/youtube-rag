from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from .database import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    youtube_video_id = Column(String, unique=True, index=True)
    title = Column(String)
    channel_name = Column(String)
    thumbnail_url = Column(String)
    url = Column(String)
    transcript_text = Column(Text)
    added_at = Column(DateTime(timezone=True), server_default=func.now())
