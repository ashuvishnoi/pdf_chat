from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from typing import List
import tempfile
import os
from core import extract_text_from_pdf, split_text, query_llm, save_to_vectorstore, send_to_slack

app = FastAPI()


@app.post("/query-pdf/")
async def query_pdf(queries: List[str] = Query(...), file: UploadFile = File(...)):
    # Validate input queries
    if not queries:
        raise HTTPException(status_code=400, detail="Queries are required")

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name
        try:
            contents = await file.read()
            temp_file.write(contents)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to save file")

    try:
        text = extract_text_from_pdf(temp_file_path)

        text_chunks = split_text(text)

        db = save_to_vectorstore(text_chunks)

        answers = query_llm(db, queries)

        # send_to_slack(answers)

        return JSONResponse(content=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.remove(temp_file_path)
