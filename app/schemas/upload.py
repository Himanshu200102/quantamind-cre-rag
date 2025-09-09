from pydantic import BaseModel

class UploadAck(BaseModel):
    ok: bool = True
    note: str = "Upload endpoint not implemented in this version."
