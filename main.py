from fastapi import FastAPI, HTTPException
from request_and_response import Request, Response
from procedure import get_sentiment

import uvicorn

app = FastAPI()


def main():
    print("Api start up!")


@app.get("")
def home():
    return "Swedish Sentiment API"


@app.post("/sentiment/", response_model=Response)
def sentiment(request_model: Request):
    try:
        result = get_sentiment(Request.input)
        return Response(sentiment=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    main()
