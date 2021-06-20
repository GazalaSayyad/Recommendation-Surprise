
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pickle
app = FastAPI()


class SvdCf(BaseModel):
    user_id: int
    movie_id: int
    rating: float

@app.get('/')
def get_root():
    return {'message': 'Welcome to the Movie Recommendation API'}

@app.post('/predict')
async def predict_rating(cf: SvdCf):
    data = cf.dict()
    loaded_model = joblib.load(open('scoresvd.pkl', 'rb'))
    prediction = loaded_model.predict(uid=data['user_id'],iid=data['movie_id'],r_ui=data['rating'])
    return {
    'prediction': prediction
    }


if __name__=='__main__':
    uvicorn.run('main:app', port=8000, host ='0.0.0.0',reload =True)
