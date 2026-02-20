from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
)
from services.pipeline_service import PipelineService


app = FastAPI(
    title="Spectral and Information-Geometric Modeling of Philosophical Assumption Structures in Text",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = PipelineService()


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse, responses={400: {"model": ErrorResponse}})
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return await run_in_threadpool(system.analyze, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# NOTE: Export endpoints (/api/export/assumptions-csv, /api/export/graph-json,
# /api/export/instability-summary) have been removed. All exports are handled
# client-side in the frontend (see App.tsx exportCsv / exportJson / exportGraphJson
# / exportSummary). Re-posting the full AnalyzeResponse body just to re-serialize
# it server-side was wasteful and those endpoints were never called by the frontend.
