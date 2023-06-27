import os
from dotenv import load_dotenv

load_dotenv()

import cloudinary
import cloudinary.uploader
import cloudinary.api

# Cloudinary API
CLOUD_NAME=os.environ.get("CLOUD_NAME")
API_KEY=os.environ.get("CLOUD_API_KEY")
API_SECRET=os.environ.get("CLOUD_API_SECRET")

# Cloudinary CONFIG
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET
)

def uploadFile(fileAudio):
    cloudinary.config()
    upload_result = None
    file_to_upload = fileAudio

    if file_to_upload:
        upload_result = cloudinary.uploader.upload(file_to_upload, resource_type="video")
        return upload_result