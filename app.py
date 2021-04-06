from typing import Optional

from fastapi import FastAPI

import uvicorn
import pickle
import numpy as np
from typing import List
from pydantic import BaseModel

class Item(BaseModel):
    sample: List[float]

# Load model
model = pickle.load(open('./models/model.pkl', 'rb'))

app = FastAPI()

@app.get("/")
def read_root():
    return {"MachineLearning2": "AdrianJimenez"}

@app.post('/predict')
async def predict(data:Item):
    result = model.predict(np.array(data.sample).reshape(1,4))
    return {"The sample belongs to class": str(result[0])}
    # return str(result[0])

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}


if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)