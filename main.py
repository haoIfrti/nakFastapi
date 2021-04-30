from fastapi import FastAPI
import uvicorn

from apis import users
from apis import passwords

app = FastAPI()
app.include_router(users.router, tags=["users"])
app.include_router(passwords.router, tags=["passwords"])


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="127.0.0.1", port=5000, reload=True, debug=True)