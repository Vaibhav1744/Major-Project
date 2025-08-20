import os
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB connection string from environment variables
mongo_uri = os.getenv("MONGO_URI")

# Initialize MongoDB client
client = pymongo.MongoClient(mongo_uri)
db = client["emotion_music_app"]
users_collection = db["users"]

# Create indexes for faster queries and uniqueness
users_collection.create_index([("email", pymongo.ASCENDING)], unique=True)
users_collection.create_index([("username", pymongo.ASCENDING)], unique=True)