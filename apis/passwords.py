from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class EditPassword(BaseModel):
    userid: int
    username: str
    old_password: str
    new_password: str
    comfirme_password: str


@router.post("/passwords/edit")
async def edit_password(item: EditPassword):
    return {"EditPassword": item}