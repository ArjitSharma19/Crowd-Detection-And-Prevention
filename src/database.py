import os
import motor.motor_asyncio
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI", "mongodb://host.docker.internal:27017/crowdshield")

# Initialize Motor async client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.get_default_database()

# Collection references
users_col = db["users"]
settings_col = db["settings"]
incidents_col = db["incidents"]

async def init_db(default_admin_password_hash: str):
    """
    Initializes collections, seeds default configurations and a default admin user if not present.
    """
    # 1. Seed Default Settings
    existing_settings = await settings_col.find_one()
    if not existing_settings:
        default_settings = {
            "max_capacity": 1000,
            "caution_at": 70,
            "trigger_delay_seconds": 20.0,
            "confidence_threshold": 0.25,
            "imgsz": 960,
            "model_type": "general",
            "detection_mode": "auto"
        }
        await settings_col.insert_one(default_settings)
        print("Database: Seeded default settings configuration.")
    else:
        print("Database: Existing settings found, skipping seeding.")

    # 2. Seed Default Admin User
    existing_admin = await users_col.find_one({"username": "admin"})
    if not existing_admin:
        default_admin = {
            "username": "admin",
            "password_hash": default_admin_password_hash,
            "role": "admin"
        }
        await users_col.insert_one(default_admin)
        print("Database: Seeded default 'admin' user.")
    else:
        print("Database: Existing 'admin' user found, skipping seeding.")

async def log_incident_to_db(zone_id: int, risk_tier: str, flow_status: str, trigger_reason: str):
    """
    Asynchronously logs a safety alert incident to the database.
    """
    incident = {
        "timestamp": datetime.utcnow(),
        "zone_id": zone_id,
        "risk_tier": risk_tier,
        "flow_status": flow_status,
        "trigger_reason": trigger_reason
    }
    await incidents_col.insert_one(incident)
    print(f"Database: Logged safety incident in Zone {zone_id} to database.")
