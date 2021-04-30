from fastapi import APIRouter


router = APIRouter()

@router.get("/users")
async def users():
    return [{"username": "user1"}, {"username": "user2"}]

@router.get("/users/{user_id}")
async def get_user_by_id(user_id: int):
    return {"userid": user_id, "username": "haoIfrit"}