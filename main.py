from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# from deta import Drive
import model
import taco_inference

app = FastAPI()

# deta = Deta("c06ow8l1_Doy4jREZEkQJj17hnXFzSM5dLf7Au9zC")
# drive = Drive("simple_drive")

# items = {
#     "foo": {"name": "Foo", "price": 50.2},
#     "bar": {"name": "Bar", "description": "The Bar fighters", "price": 62, "tax": 20.2},
#     "baz": {
#         "name": "Baz",
#         "description": "There goes my baz",
#         "price": 50.2,
#         "tax": 10.5,
#     },
# }

fake_items_db = [
    {"item_name": "Foo"},
    {"item_name": "Bar"},
    {"item_name": "Baz"}]

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

class Item(BaseModel):
    title: str
    # timestamp: datetime
    # description: Union[str, None] = None

@app.get("/")
def mainmain():
    tacotacotaco = 1
    # tacotacotaco = taco_inference.taco_config()
    return tacotacotaco

@app.get("/aaitems/")
def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]



# @app.get("/items/{id}",json={items})
# def update_item(id: str):
#     json_compatible_item_data = jsonable_encoder(Item)
#     return JSONResponse(content=json_compatible_item_data)

# @app.get("/{ssss}")
# def mainmain(ssss: int):
#     return ssss

# class Item(BaseModel):
#     title: str
#     timestamp: datetime
#     description: Union[str, None] = None



# @app.get(
#     "/items/{item_id}",
#     response_model=Item,
#     response_model_exclude_unset=True,
#     response_model_include=["name", "description"],
#     response_model_exclude=["tax"]
# )
# async def read_item(item_id: str):
#     return items[item_id]

# @app.get("/user/{user}", response_model=UserIn)
# def read_item(user: str):
#     print(user)
#     return user
#     return FileResponse('index.html')

# async def returnqq(item_id: str, request: Request):
# @app.put("/{id}")
# async def returnqq(id: str, item: Item):
#     json_compatible_item_data = jsonable_encoder(item)
#     return JSONResponse(content=json_compatible_item_data)

# @app.get("/{item_id}")
# def mainmain(item_id: int):
#     return {"item_id" : item_id}

# @app.get("/user")
# def roofa():
#     conn = pymysql.connect(
#         host='192.168.0.16',
#         user='better',
#         password='q1w2e3r41!',
#     )
#         # db='better',
#         # charset='utf8')
#     cursor = conn.cursor()
#     cursor.execute(
#         """
#         use better;
#         """
#         )
#     cursor.execute(
#     """
#     select * from better_user;
#     """
#     )
#     # cursor.execute(
#     #     """
#     #     insert into better_user (id, password, name) values("id", "password", "name");
#     #     """
#     # )
#     # conn.commit()
#     a = cursor.fetchall()
#     conn.close()
#     return a


# from pydantic import BaseModel
    
# class Model(BaseModel):
#     name : str
#     phone : int

# # @app.get("/db")
# # def roofie(data: Model):
# #     print("fsads")
# #     conn = pymysql.connect(host='192.168.56.1', user='root', password='yll10!', db='better', charset='utf8')
# #     # if conn:
# #     #     print("daasdfasdf")
# #     # else: print("tqwr")
# #     cur = conn.cursor()
# #     cur.execute('insert into better_user values("Park", 0000, "ㅇㅇ")')
# #     conn.commit()
# #     return FileResponse('index.html')