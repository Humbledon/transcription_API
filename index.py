from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import time as time_module
from datetime import timedelta
from faster_whisper import WhisperModel
import shutil

app = FastAPI()

# Initialize Whisper model
model_size = "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    print("Transcribing audio...")
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add timer at the start
        start_time = time_module.time()
        
        # Transcribe audio
        segments, info = model.transcribe(
            temp_file, 
            beam_size=5,
            language="fr",
            condition_on_previous_text=True
        )
        
        # Process segments
        transcription = []
        segments_list = list(segments)
        
        if not segments_list:
            return JSONResponse(
                status_code=400,
                content={"error": "Aucun segment trouvé!"}
            )
            
        for segment in segments_list:
            if hasattr(segment, 'text') and segment.text:
                text = segment.text.strip()
                if text:
                    transcription.append({
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": text
                    })
        
        # Calculate execution time
        execution_time = time_module.time() - start_time
        
        # Clean up temp file
        os.remove(temp_file)
        
        return {
            "execution_time": str(timedelta(seconds=execution_time)),
            "transcription": transcription
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur pendant la transcription: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)