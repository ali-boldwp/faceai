from pydantic import BaseModel

class ImageURLs(BaseModel):
    front_image_url: str
    side_image_url: str
