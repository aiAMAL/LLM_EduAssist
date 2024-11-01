import os
import sys
import uvicorn
from fastapi import FastAPI
from starlette.responses import Response, RedirectResponse
# from src.TextSummarization.pipeline.pipeline_prediction import PredictionPipeline
from src.TextSummarization.pipeline.langchain_prediction import PredictionPipeline

app = FastAPI()


@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')


@app.get('/train')
async def train_model():
    try:
        os.system('python main.py')
        return Response('Training completed successfully')
    except Exception as e:
        return Response(f'Error occurred during training the model: {str(e)}')


@app.get("/predict")
async def predict_route(text: str):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return summary
    except Exception as e:
        raise e


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
    # uvicorn.run(app, host='0.0.0.0', port=8000)

